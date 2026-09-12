import numpy as np

def export_onnx(model, input_shape, path="model.onnx", model_name="flux_model"):
    """
    Export a flux Sequential model to ONNX format.
    Supports: Linear, ReLU, Sigmoid, Tanh, LayerNorm, Dropout (eval).
    """
    import struct

    def _tensor_proto(name, data):
        raw = data.astype(np.float32).tobytes()
        shape = list(data.shape)
        return (name, shape, raw)

    initializers = []
    nodes = []
    input_name = "input"
    current = input_name
    node_idx = 0

    from flux.nn.linear import Linear
    from flux.nn.activations import ReLU, Sigmoid, Tanh
    from flux.nn.normalization import LayerNorm, Dropout
    from flux.nn.sequential import Sequential

    layers = model.layers if isinstance(model, Sequential) else [model]

    for layer in layers:
        out_name = f"node_{node_idx}"
        if isinstance(layer, Linear):
            w_name = f"weight_{node_idx}"
            b_name = f"bias_{node_idx}"
            initializers.append(_tensor_proto(w_name, layer.weight.data.T.astype(np.float32)))
            if layer.bias is not None:
                initializers.append(_tensor_proto(b_name, layer.bias.data.astype(np.float32)))
                nodes.append(("Gemm", [current, w_name, b_name], [out_name], {}))
            else:
                nodes.append(("Gemm", [current, w_name], [out_name], {"transB": 0}))
        elif isinstance(layer, ReLU):
            nodes.append(("Relu", [current], [out_name], {}))
        elif isinstance(layer, Sigmoid):
            nodes.append(("Sigmoid", [current], [out_name], {}))
        elif isinstance(layer, Tanh):
            nodes.append(("Tanh", [current], [out_name], {}))
        elif isinstance(layer, LayerNorm):
            g_name = f"ln_gamma_{node_idx}"
            b_name2 = f"ln_beta_{node_idx}"
            initializers.append(_tensor_proto(g_name, layer.gamma.data.astype(np.float32)))
            initializers.append(_tensor_proto(b_name2, layer.beta.data.astype(np.float32)))
            nodes.append(("LayerNormalization", [current, g_name, b_name2], [out_name],
                          {"epsilon": float(layer.eps)}))
        elif isinstance(layer, Dropout):
            nodes.append(("Identity", [current], [out_name], {}))
        else:
            raise ValueError(f"Unsupported layer for ONNX export: {type(layer)}")
        current = out_name
        node_idx += 1

    # write minimal ONNX protobuf manually
    def _vi(name, shape):
        # ValueInfoProto: name + type (tensor float32) + shape
        return (name, shape)

    graph_input  = [(input_name, list(input_shape))]
    graph_output = [(current, None)]

    _write_onnx_proto(path, model_name, initializers, nodes, graph_input, graph_output)
    print(f"Exported to {path}")
    return path


def _write_onnx_proto(path, name, initializers, nodes, inputs, outputs):
    """Write a minimal valid ONNX file using raw protobuf encoding."""
    import struct

    def _varint(n):
        buf = []
        while n > 0x7F:
            buf.append((n & 0x7F) | 0x80)
            n >>= 7
        buf.append(n)
        return bytes(buf)

    def _field(field_num, wire_type, data):
        tag = (field_num << 3) | wire_type
        return _varint(tag) + data

    def _bytes_field(field_num, data):
        return _field(field_num, 2, _varint(len(data)) + data)

    def _string(s):
        b = s.encode("utf-8")
        return _varint(len(b)) + b

    def _string_field(field_num, s):
        return _bytes_field(field_num, s.encode("utf-8"))

    def _int64_field(field_num, n):
        # wire type 0 = varint
        tag = _varint((field_num << 3) | 0)
        return tag + _varint(n if n >= 0 else n + (1 << 64))

    def _float_field(field_num, v):
        tag = _varint((field_num << 3) | 5)
        return tag + struct.pack("<f", v)

    def _tensor_proto_bytes(name, shape, raw_data):
        """TensorProto"""
        b = b""
        b += _int64_field(1, 1)  # data_type = FLOAT
        for s in shape:
            b += _int64_field(3, s)  # dims
        b += _bytes_field(9, raw_data)  # raw_data
        b += _string_field(8, name)  # name
        return b

    def _type_proto_tensor(shape):
        """TypeProto with tensor of float32"""
        shape_proto = b""
        if shape:
            for d in shape:
                dim = b""
                if d is not None:
                    dim += _int64_field(1, d)
                shape_proto += _bytes_field(1, dim)
        tensor_type = _int64_field(1, 1)  # elem_type=FLOAT
        if shape:
            tensor_type += _bytes_field(2, shape_proto)
        type_proto = _bytes_field(1, tensor_type)
        return type_proto

    def _value_info(name, shape):
        b = _string_field(1, name)
        b += _bytes_field(2, _type_proto_tensor(shape))
        return b

    def _attr(key, val):
        b = _string_field(1, key)
        if isinstance(val, float):
            b += _float_field(4, val)
            b += _int64_field(20, 1)  # FLOAT type
        elif isinstance(val, int):
            b += _int64_field(3, val)
            b += _int64_field(20, 2)  # INT type
        return b

    def _node_proto(op, inputs, outputs, attrs):
        b = b""
        for i in inputs:
            b += _string_field(1, i)
        for o in outputs:
            b += _string_field(2, o)
        b += _string_field(4, op)
        b += _string_field(7, "flux")
        for k, v in attrs.items():
            b += _bytes_field(5, _attr(k, v))
        return b

    # Build graph
    graph = b""
    # nodes
    for op, inp, out, attrs in nodes:
        graph += _bytes_field(1, _node_proto(op, inp, out, attrs))
    graph += _string_field(2, name)
    # inputs
    for iname, ishape in inputs:
        graph += _bytes_field(11, _value_info(iname, ishape))
    # outputs
    for oname, oshape in outputs:
        graph += _bytes_field(12, _value_info(oname, oshape or []))
    # initializers
    for tname, tshape, traw in initializers:
        graph += _bytes_field(5, _tensor_proto_bytes(tname, tshape, traw))

    # Build model
    model_proto = b""
    model_proto += _int64_field(1, 8)        # ir_version=8
    model_proto += _bytes_field(7, graph)    # graph
    model_proto += _string_field(2, "flux")  # domain
    model_proto += _int64_field(5, 17)       # opset_version=17
    # opset_import
    opset = _int64_field(2, 17)
    model_proto += _bytes_field(8, opset)

    with open(path, "wb") as f:
        f.write(model_proto)
