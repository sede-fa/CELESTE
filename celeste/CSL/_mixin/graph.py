import networkx as nx


class GraphMixin:
    """Mermaid and NetworkX graph representations of the system topology."""

    def makeGraph(self) -> None:
        """
        Will be replaced with create_graph()
        Generate graph representation of the system using Mermaid syntax.
        """
        self.create_graph("mermaid")

    def make_mermaid_graph(self, save: bool | str = True) -> str:
        """
        Create a Mermaid graph representation of the system.

        Args:
            save: ``True`` writes ``<system name>.mmd`` into the working directory (default),
                  a string is used as the file path, ``False`` writes nothing.

        Returns:
            str: Mermaid syntax for the graph.
        """
        mermaid_lines = ["flowchart LR\n"]

        for obj in self.getObjs():
            if obj != self:
                mermaid_lines.append(f"{obj.name}\n")

        for connection in self.connections:
            mermaid_lines.append(f"\t{connection[0]} --{connection[1]}_{connection[2]}--> {connection[3]}\n")

        text = "".join(mermaid_lines)
        if save:
            path = save if isinstance(save, str) else f'{self.name}.mmd'
            with open(path, 'w') as f:
                f.write(text)
        return text

    def make_networkx_graph(self) -> nx.DiGraph:
        """
        Create a NetworkX graph representation of the system.

        Returns:
            nx.DiGraph: A directed graph representing the system's components and connections.
        """
        import matplotlib.pyplot as plt
        G = nx.DiGraph()

        for obj in self.getObjs():
            if obj != self:
                G.add_node(obj.name, type=obj.__class__.__name__, attributes=obj.__dict__)

        edge_labels = {}
        for connection in self.connections:
            source, target, label = connection[0], connection[-1], f"{connection[1]}_{connection[2]}"
            G.add_edge(source, target)
            edge_labels[(source, target)] = label

        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
        return G

    def create_graph(self, graph_type: str = "mermaid", save: bool | str = True) -> nx.DiGraph | str:
        """
        Create a graph representation of the system based on the specified type.

        Args:
            graph_type (str): The type of graph to create ("networkx" or "mermaid").
            save: for "mermaid": write the graph to ``<system name>.mmd`` (True), to a given path (str) or not at all (False).

        Returns:
            Union[nx.DiGraph, str]: The generated graph (NetworkX object or Mermaid syntax).

        Raises:
            ValueError: If an unsupported graph type is specified.
        """
        self.connections = []
        self.seq_sol(store=True)
        if graph_type.lower() == "networkx":
            return self.make_networkx_graph()
        elif graph_type.lower() == "mermaid":
            return self.make_mermaid_graph(save=save)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}. Choose 'networkx' or 'mermaid'.")
