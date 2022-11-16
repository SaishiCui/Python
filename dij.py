graph={}

with open("C:/Users/cuisa/Desktop/Dij.txt") as f:
    data = f.readlines()
    for line in data:
        elements = list(map(str, line.split('\t')[:-1] ) )
        graph[str(elements[0])]=elements[1:]
f.close()

print(graph.values())


shortest_path={}

def dijkstra(graph, Node):
    global shortest_path
    shortest_path[Node]=0
    growing_node = {Node}
    while (len(growing_node) != len(graph)):
        mini=1000000
        mini_edge=(None, None)
        for node in growing_node:
            for edge in graph[node]:
                head_node = edge.split(",")[0]
                length = int(edge.split(",")[1])
                if head_node not in growing_node:
                    if shortest_path[node]+length < mini:
                        mini_edge=(node, head_node)
                        mini = shortest_path[node]+length
        if mini_edge != (None, None):
            growing_node.add(mini_edge[1])
            shortest_path[mini_edge[1]]=mini
        else:
            for key in graph.keys():
                if key not in growing_node:
                    growing_node.add(key)
                    shortest_path[key]=mini

dijkstra(graph , "1")
                
    
ans = ''
for i in ['7','37','59','82','99','115','133','165','188','197']:
    ans += str(shortest_path[i]) + ","

ans = ans[:-1]
print(ans)