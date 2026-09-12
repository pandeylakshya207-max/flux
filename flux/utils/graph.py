def build_dot(tensor, rankdir="LR"):
    """Return a Graphviz DOT string for the computation graph."""
    nodes, edges = set(), set()

    def walk(t):
        if id(t) in nodes:
            return
        nodes.add(id(t))
        label = f"data={t.data.flat[0]:.3f}" if t.data.size == 1 else f"shape={tuple(t.data.shape)}"
        if t._op:
            op_id = id(t) + 1
            nodes.add(op_id)
            edges.add((op_id, id(t)))
            for child in t._prev:
                edges.add((id(child), op_id))
                walk(child)

    walk(tensor)

    lines = [f'digraph G {{ rankdir={rankdir}']
    visited_ops = set()
    def walk2(t):
        if id(t) in visited_ops:
            return
        visited_ops.add(id(t))
        label = f"data={t.data.flat[0]:.3f}" if t.data.size == 1 else f"shape={tuple(t.data.shape)}"
        grad_str = ""
        if t.grad is not None:
            g = t.grad.flat[0] if t.grad.size == 1 else "..."
            grad_str = f"\\ngrad={g:.3f}" if isinstance(g, float) else f"\\ngrad={g}"
        lines.append(f'  {id(t)} [label="{label}{grad_str}" shape=box]')
        if t._op:
            op_id = id(t) + 1
            lines.append(f'  {op_id} [label="{t._op}" shape=ellipse]')
            lines.append(f'  {op_id} -> {id(t)}')
            for child in t._prev:
                lines.append(f'  {id(child)} -> {op_id}')
                walk2(child)
    walk2(tensor)
    lines.append("}")
    return "\n".join(lines)

def save_dot(tensor, path="graph.dot"):
    dot = build_dot(tensor)
    with open(path, "w") as f:
        f.write(dot)
    print(f"Saved graph to {path}")
    return dot
