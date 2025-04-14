"""Generate a direct colored DOT representation of the torch_sim module dependencies.

This script analyzes the torch_sim package structure by directly inspecting imports
and creates a DOT file with:
1. Nodes colored by their connectedness (number of connections)
2. No "torch_sim." prefix in node labels
3. Line count information for each module
4. Customizable layout parameters via command-line options
"""

import argparse
import ast
import colorsys
import os
import tomllib
from collections import defaultdict


with open("pyproject.toml", "rb") as pyproject:
    github_base_url = tomllib.load(pyproject)["project"]["urls"]["Repo"]


def rgb_to_hex(red: float, green: float, blue: float) -> str:
    """Convert RGB values to hex color code."""
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def generate_heat_colors(num_colors: int) -> list:
    """Generate a color gradient from blue (cool) to red (hot)."""
    colors = []
    for index in range(num_colors):
        # Hue range from blue (0.66) to red (0)
        hue = 0.66 * (1 - index / (num_colors - 1)) if num_colors > 1 else 0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb_to_hex(red, green, blue))
    return colors


def get_file_imports(file_path: str) -> list:
    """Extract all import statements from a Python file."""
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            tree = ast.parse(file_handle.read())

        imports = []

        # Extract regular imports (import x, import y)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(name.name for name in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
    except SyntaxError:
        print(f"Syntax error in {file_path}")
        return []
    else:
        return imports


def count_lines(file_path: str) -> int:
    """Count non-empty, non-comment lines in a Python file."""
    with open(file_path, encoding="utf-8") as file_handle:
        return sum(
            bool(line.strip() and not line.strip().startswith("#"))
            for line in file_handle
        )


def analyze_package(
    package_path: str, package_name: str = "torch_sim"
) -> dict[str, set[str]]:
    """Build a dependency graph of package modules."""
    dependency_graph: defaultdict[str, set[str]] = defaultdict(set)

    # Get Python files excluding those starting with _
    python_files = [
        f"{root}/{file}"
        for root, _, files in os.walk(package_path)
        for file in files
        if file.endswith(".py") and not file.startswith("_")
    ]

    print(f"Found {len(python_files)} Python files in {package_path}")

    for file_path in python_files:
        # Convert file path to module name
        rel_path = os.path.relpath(file_path, os.path.dirname(package_path))
        module_name = rel_path.replace(".py", "").replace("/", ".").replace("\\", ".")

        # Get imports and add only torch_sim imports to the graph
        imports = get_file_imports(file_path)
        torch_sim_imports = [
            import_name
            for import_name in imports
            if import_name == package_name or import_name.startswith(f"{package_name}.")
        ]

        for imported_module in torch_sim_imports:
            dependency_graph[module_name].add(imported_module)

    return dependency_graph


def simplify_module_name(full_name: str, base_package: str = "torch_sim") -> str:
    """Remove the base package prefix from module names."""
    if full_name == base_package:
        return base_package
    if full_name.startswith(f"{base_package}."):
        return full_name[len(base_package) + 1 :]
    return full_name


def module_to_node_id(module_name: str) -> str:
    """Convert module name to valid DOT node ID."""
    return module_name.replace(".", "_")


def generate_dot_file(  # noqa: C901, PLR0915
    dependency_graph: dict[str, set[str]],
    output_file: str,
    package_path: str,
    args: argparse.Namespace,
) -> None:
    """Generate a DOT file with connectedness-based coloring."""
    connections: defaultdict[str, int] = defaultdict(int)
    all_modules = set(dependency_graph.keys()).union(
        {dep for deps in dependency_graph.values() for dep in deps}
    )

    # Count connections (outgoing + incoming)
    for module, deps in dependency_graph.items():
        connections[module] += len(deps)  # Outgoing
        for dep in deps:
            if dep in all_modules:
                connections[dep] += 1  # Incoming

    # Define connection ranges and colors
    connection_ranges = ["0-1", "2-3", "4-5", "6-7", "8-10", "11-15", "16+"]
    range_thresholds = [1, 3, 5, 7, 10, 15, float("inf")]  # Upper bounds for each range
    colors = generate_heat_colors(len(connection_ranges))

    # Map connection counts to ranges
    def get_range(count: int) -> str:
        for idx, threshold in enumerate(range_thresholds):
            if count <= threshold:
                return connection_ranges[idx]
        return connection_ranges[-1]  # Should never reach here

    # Start generating DOT content
    lines = ["digraph G {", "    layout=dot;"]

    # Add layout parameters
    if args.engine != "dot":
        lines.append(f"    layout = {args.engine};")

    lines += [f"    concentrate = {str(args.concentrate).lower()};"]
    lines += [
        f"    {key} = {getattr(args, key)};"
        for key in "ratio nodesep ranksep rankdir overlap splines maxiter".split()  # noqa: SIM905
        if getattr(args, key, None) is not None
    ]

    if args.pack:
        lines.append("    pack = true;")

    # Engine-specific options
    if args.engine == "fdp":
        lines.append(f"    K = {args.K};")
    elif args.engine == "neato":
        lines.append("    epsilon = 0.00001;")

    # Node styling
    node_attrs = [
        "style=filled",
        'fillcolor="#ffffff"',
        'fontcolor="#000000"',
        "fontname=Helvetica",
        "fontsize=10",
        f'margin="{args.margin}"',
    ]
    if args.node_height:
        node_attrs.append(f"height={args.node_height}")

    lines.append(f"    node [{','.join(node_attrs)}];")
    lines.append("")

    # Add color legend
    lines.append("    // Color legend by node connectedness")
    range_to_color = {
        range_name: colors[idx] for idx, range_name in enumerate(connection_ranges)
    }
    for range_name in connection_ranges:
        lines.append(f"    // {range_to_color[range_name]} = {range_name} connections")
    lines.append("")

    # Add nodes
    for module in sorted(all_modules):
        simple_name = simplify_module_name(module)
        node_id = module_to_node_id(module)
        range_name = get_range(connections[module])
        color = range_to_color[range_name]

        # Count lines if it's a torch_sim module
        label = simple_name
        github_url = ""

        if module.startswith("torch_sim"):
            relative_module = module.replace(".", os.sep)
            module_name = relative_module[len("torch_sim") + 1 :]
            module_path = f"{package_path}/{module_name}.py"
            # Include 'torch_sim' in the GitHub URL path
            github_url = f"{github_base_url}/blob/main/torch_sim/{module_name}.py"

            if os.path.isfile(module_path):
                line_count = count_lines(module_path)
                label = f"{simple_name}\\n({line_count} lines)"

        node_style = f'fillcolor="{color}",fontcolor="white",label="{label}",shape="box"'
        if github_url:
            node_style += f',URL="{github_url}",tooltip="View source on GitHub"'

        lines.append(f"    {node_id} [{node_style}];")

    # Add edges
    lines.append("")
    for module, deps in sorted(dependency_graph.items()):
        source_id = module_to_node_id(module)
        for dep in sorted(deps):
            if dep in all_modules:
                target_id = module_to_node_id(dep)
                lines.append(f"    {source_id} -> {target_id};")

    lines.append("}")

    # Write to file
    with open(output_file, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")

    print(f"Generated DOT file: {output_file}")

    # Print statistics
    print("\nNode connectedness statistics:")
    stats = defaultdict(list)
    for module, count in connections.items():
        stats[get_range(count)].append((module, count))

    for range_name in connection_ranges:
        if nodes := stats.get(range_name):
            avg = sum(node[1] for node in nodes) / len(nodes)
            print(f"  {range_name}: {len(nodes)} nodes, avg {avg:.1f} connections")
            if range_name in ["11-15", "16+"]:
                hub_modules = ", ".join(simplify_module_name(node[0]) for node in nodes)
                print(f"    - Hub modules: {hub_modules}")


def main() -> None:
    """Entry point that parses args and generates the DOT file."""
    parser = argparse.ArgumentParser(
        description="Generate a colored DOT file of torch_sim module dependencies"
    )

    # Graph layout options
    parser.add_argument(
        "--engine",
        choices=["dot", "neato", "fdp", "circo", "twopi", "osage"],
        default="dot",
        help="GraphViz layout engine (default: dot)",
    )
    parser.add_argument(
        "--concentrate",
        action="store_true",
        default=True,
        help="Concentrate edges (default: True)",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8, help="Aspect ratio (default: 0.8)"
    )
    parser.add_argument(
        "--nodesep",
        type=float,
        default=0.08,
        help="Horizontal node separation (default: 0.08)",
    )
    parser.add_argument(
        "--ranksep",
        type=float,
        default=0.1,
        help="Vertical rank separation (default: 0.1)",
    )
    parser.add_argument(
        "--overlap",
        choices=["true", "false", "scale", "compress", "vpsc", "prism", "none"],
        default="false",
        help="Overlap handling (default: false)",
    )
    parser.add_argument(
        "--splines",
        choices=["true", "false", "ortho", "curved", "line", "polyline", "none"],
        default=None,
        help="Edge spline style (default: none)",
    )
    parser.add_argument(
        "--margin", default="0.08,0.02", help="Node margin (default: 0.08,0.02)"
    )
    parser.add_argument(
        "--node-height", type=float, default=0.5, help="Fixed node height (default: 0.5)"
    )
    parser.add_argument(
        "--rankdir",
        choices=["TB", "LR", "BT", "RL"],
        default="LR",
        help="Rank direction (default: LR)",
    )

    # Advanced options
    parser.add_argument(
        "--pack",
        action="store_true",
        default=False,
        help="Pack graph components tightly (default: False)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Max iterations for force-directed layouts",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        default=False,
        help="Enable compact layout preset",
    )
    parser.add_argument(
        "--K",
        type=float,
        default=0.1,
        help="Spring constant for force-directed layouts (default: 0.1)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        default="docs/_static",
        help="Output directory (default: docs/_static)",
    )
    parser.add_argument(
        "--output-file",
        default="torch-sim-module-graph.dot",
        help="Output filename (default: torch-sim-module-graph.dot)",
    )

    args, _unknown = parser.parse_known_args()

    # Apply compact preset if selected
    if args.compact:
        args.rankdir = args.rankdir or "LR"
        if args.engine == "dot":
            args.ranksep, args.nodesep, args.overlap, args.splines = (
                0.01,
                0.05,
                "compress",
                "ortho",
            )
            args.node_height = args.node_height or 0.1
        elif args.engine in ["fdp", "neato"]:
            args.overlap, args.pack, args.maxiter = "prism", True, 500

    # Set up paths
    package_name = "torch_sim"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for the package at the project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    package_path = f"{project_root}/{package_name}"

    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{args.output_file}"

    print(f"Analyzing package: {package_name} at {package_path}")

    dependency_graph = analyze_package(package_path, package_name)
    generate_dot_file(dependency_graph, output_file, package_path, args)

    print(f"\nDone! Generated DOT file: {output_file}")
    print(f"Layout engine: {args.engine}")
    print("\nTo render directly with GraphViz (if installed):")
    svg_path = output_file.replace(".dot", ".svg")
    print(f"  dot -T{args.engine} -Tsvg {output_file} -o {svg_path}")


if __name__ == "__main__":
    main()
