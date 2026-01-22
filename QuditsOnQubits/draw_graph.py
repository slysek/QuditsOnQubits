from igraph import Graph, plot
import matplotlib.pyplot as plt

def draw_graph(graph):
    graph.vs["name"] = [i + 1 for i in range(graph.vcount())]
    fig, ax = plt.subplots()
    plot(graph, target=ax, vertex_label=graph.vs["name"])