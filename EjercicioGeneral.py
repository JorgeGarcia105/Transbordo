import pulp as op
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- DATOS DEL PROBLEMA (TechHardware Corp.) ---

# Nodos de oferta (orígenes)
nodos_oferta = ["I1", "I2", "I3", "I4", "I5", "I6", "I7"]
nombres_oferta = ["I1", "I2", "I3", "I4", "I5", "I6", "I7"]

# Nodos de transbordo
nodos_transbordo = ["J1", "J2", "J3", "J4"]
nombres_transbordo = ["J1", "J2", "J3", "J4"]

# Nodos de demanda (destinos)
nodos_demanda = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"]
nombres_demanda = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"]

# Todos los nodos
nodos = nodos_oferta + nodos_transbordo + nodos_demanda

# Arcos (relaciones entre nodos)
arcos = [
    # Oferta-Transbordo
    ("I1", "J1"), ("I1", "J2"), ("I1", "K1"),
    ("I2", "J1"), ("I2", "J4"),
    ("I3", "J2"), ("I3", "J3"),
    ("I4", "J1"), ("I4", "J2"), ("I4", "J4"),
    ("I5", "J3"),
    ("I6", "J3"), ("I6", "J4"),
    ("I7", "J1"), ("I7", "J2"), ("I7", "K1"),
    # Transbordo-Demanda
    ("J1", "K1"), ("J1", "K4"), ("J1", "K5"), ("J1", "K7"), ("J1", "K8"), ("J1", "K10"),
    ("J2", "K4"), ("J2", "K6"), ("J2", "K7"), ("J2", "K10"),
    ("J3", "K1"), ("J3", "K2"), ("J3", "K3"), ("J3", "K5"), ("J3", "K9"),
    ("J4", "K1"), ("J4", "K5"), ("J4", "K6"),
    # Oferta-Demanda
    ("I1", "K1"), ("I7", "K1")
]

# Oferta en cada nodo de oferta
oferta = {
    "I1": 300,
    "I2": 400,
    "I3": 360,
    "I4": 200,
    "I5": 380,
    "I6": 230,
    "I7": 400,
}

# Demanda en cada nodo de demanda
demanda = {
    "K1": 120,
    "K2": 100,
    "K3": 80,
    "K4": 100,
    "K5": 140,
    "K6": 200,
    "K7": 60,
    "K8": 90,
    "K9": 90,
    "K10": 100,
}

# Costos de transporte por unidad en cada arco
costos = {
    ("I1", "J1"): 230, ("I1", "J2"): 300, ("I1", "K1"): 290,
    ("I2", "J1"): 250, ("I2", "J4"): 280,
    ("I3", "J2"): 325, ("I3", "J3"): 315,
    ("I4", "J1"): 190, ("I4", "J2"): 225, ("I4", "J4"): 325,
    ("I5", "J3"): 305,
    ("I6", "J3"): 360, ("I6", "J4"): 310,
    ("I7", "J1"): 270, ("I7", "J2"): 340, ("I7", "K1"): 190,
    ("J1", "K1"): 350, ("J1", "K4"): 290, ("J1", "K5"): 310, ("J1", "K7"): 250, ("J1", "K8"): 320, ("J1", "K10"): 370,
    ("J2", "K4"): 330, ("J2", "K6"): 260, ("J2", "K7"): 280, ("J2", "K10"): 335,
    ("J3", "K1"): 370, ("J3", "K2"): 190, ("J3", "K3"): 305, ("J3", "K5"): 330, ("J3", "K9"): 260,
    ("J4", "K1"): 190, ("J4", "K5"): 305, ("J4", "K6"): 345,
    ("I1", "K1"): 290, ("I7", "K1"): 190
}

# --- MODELO DE OPTIMIZACIÓN ---

prob = op.LpProblem("ProblemaDeTransporte_TechHardware", op.LpMinimize)

# Variables de decisión (flujo en cada arco)
x = op.LpVariable.dicts("x", arcos, lowBound=0, cat=op.LpContinuous)

# Función objetivo: minimizar costos de transporte
prob += op.lpSum([costos[i, j] * x[i, j] for (i, j) in arcos])

# Restricciones
for i in nodos:
    if i in nodos_oferta:
        prob += op.lpSum([x[i, j] for j in nodos if (i, j) in arcos]) <= oferta[i]  # Restricción de oferta
    elif i in nodos_demanda:
        prob += op.lpSum([x[j, i] for j in nodos if (j, i) in arcos]) == demanda[i]  # Restricción de demanda
    else:  # Nodos de transbordo
        prob += op.lpSum([x[i, j] for j in nodos if (i, j) in arcos]) == op.lpSum([x[j, i] for j in nodos if (j, i) in arcos])  # Restricción de balance

# Resolver el problema
prob.solve()

# --- RESULTADOS ---
print("Estado:", op.LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Valor de la funcion objetivo (Z) =", op.value(prob.objective))

# Convertir solución a un formato utilizable para NetworkX
edges = {}
for v in prob.variables():
    if v.varValue > 0:
        # Extraer nodos del nombre de la variable
        nombre = v.name
        match = re.search(r"x_\('(.*?)',\s*'(.*?)'\)", nombre)
        if match:
            nodo_origen, nodo_destino = match.groups()
            edges[(nodo_origen, nodo_destino)] = v.varValue

# --- VISUALIZACIÓN CON NETWORKX ---
# Crear el grafo con NetworkX
G = nx.DiGraph()

# Añadir aristas al grafo
for (nodo_origen, nodo_destino), peso in edges.items():
    G.add_edge(nodo_origen, nodo_destino, weight=peso)

# Posiciones predefinidas para la visualización
pos = {
    "I1": (-1, 1), "I2": (-1, 0.5), "I3": (-1, 0), "I4": (-1, -0.5), "I5": (-1, -1), "I6": (-1, -1.5), "I7": (-1, -2),
    "J1": (0, 1), "J2": (0, 0.5), "J3": (0, 0), "J4": (0, -0.5),
    "K1": (1, 1), "K2": (1, 0.5), "K3": (1, 0), "K4": (1, -0.5), "K5": (1, -1), "K6": (1, -1.5), "K7": (1, -2), "K8": (1, -2.5), "K9": (1, -3), "K10": (1, -3.5)
}

# --- VISUALIZACIÓN CON NETWORKX ---
# Crear el grafo con NetworkX
G = nx.DiGraph()

# Añadir aristas al grafo
for (nodo_origen, nodo_destino), peso in edges.items():
    G.add_edge(nodo_origen, nodo_destino, weight=peso)

# Posiciones predefinidas para la visualización
pos = {
    "I1": (-1, 1), "I2": (-1, 0.5), "I3": (-1, 0), "I4": (-1, -0.5), "I5": (-1, -1), "I6": (-1, -1.5), "I7": (-1, -2),
    "J1": (0, 1), "J2": (0, 0.5), "J3": (0, 0), "J4": (0, -0.5),
    "K1": (1, 1), "K2": (1, 0.5), "K3": (1, 0), "K4": (1, -0.5), "K5": (1, -1), "K6": (1, -1.5), "K7": (1, -2), "K8": (1, -2.5), "K9": (1, -3), "K10": (1, -3.5)
}

# Establecer la figura
plt.figure(figsize=(14, 10))
plt.gca().set_facecolor('white')  # Fondo blanco para el grafo

# Colores mejorados para los nodos
color_oferta = 'lightsalmon'
color_transbordo = 'lightblue'
color_demanda = 'lightgreen'

# Dibujar nodos con colores mejorados y bordes contrastantes
nx.draw_networkx_nodes(G, pos, nodelist=nodos_oferta, node_color=color_oferta, node_size=800, edgecolors='black', linewidths=1.5)
nx.draw_networkx_nodes(G, pos, nodelist=nodos_transbordo, node_color=color_transbordo, node_size=800, edgecolors='black', linewidths=1.5)
nx.draw_networkx_nodes(G, pos, nodelist=nodos_demanda, node_color=color_demanda, node_size=800, edgecolors='black', linewidths=1.5)

# Cambiar los colores de las aristas según el tipo de conexión
edge_colors_oferta = 'darkred'  # Color para los arcos de oferta
edge_colors_transbordo = 'royalblue'  # Color para los arcos de transbordo
edge_colors_demanda = 'green'  # Color para los arcos de demanda

# Colores claros para los arcos según el tipo de nodo
# Colores claros para los arcos según el tipo de nodo
edge_colors_oferta = 'lightcoral'  # Color claro para los arcos de oferta
edge_colors_transbordo = 'lightblue'  # Color claro para los arcos de transbordo
edge_colors_demanda = 'lightgreen'  # Color claro para los arcos de demanda

# Dibujar arcos con colores diferenciados según el tipo de nodo
for i, nodo in enumerate(nodos_oferta):
    arcos_oferta = [(nodo, destino) for destino in nodos if (nodo, destino) in arcos]
    nx.draw_networkx_edges(G, pos, edgelist=arcos_oferta, edge_color=edge_colors_oferta, width=2.5, alpha=0.7)

for i, nodo in enumerate(nodos_transbordo):
    arcos_transbordo = [(nodo, destino) for destino in nodos if (nodo, destino) in arcos]
    nx.draw_networkx_edges(G, pos, edgelist=arcos_transbordo, edge_color=edge_colors_transbordo, width=2.5, alpha=0.7)

for i, nodo in enumerate(nodos_demanda):
    arcos_demanda = [(nodo, destino) for destino in nodos if (nodo, destino) in arcos]
    nx.draw_networkx_edges(G, pos, edgelist=arcos_demanda, edge_color=edge_colors_demanda, width=2.5, alpha=0.7)




# **Líneas rectas específicas para I1-K1 y I7-K7** (destacar estas líneas)
nx.draw_networkx_edges(G, pos, edgelist=[("I1", "K1")], edge_color='darkorange', width=3, alpha=0.9, connectionstyle='arc3,rad=-0.2')
nx.draw_networkx_edges(G, pos, edgelist=[("I7", "K7")], edge_color='darkorange', width=3, alpha=0.9, connectionstyle='arc3,rad=0.0')

# Etiquetas de los arcos con valores de peso
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, font_color='black')

# Dibujar etiquetas de los nodos con una fuente legible
labels = {nodo: nombres_oferta[i] for i, nodo in enumerate(nodos_oferta)}
labels.update({nodo: nombres_transbordo[i] for i, nodo in enumerate(nodos_transbordo)})
labels.update({nodo: nombres_demanda[i] for i, nodo in enumerate(nodos_demanda)})
nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', font_color='black')

# Mejorar la visibilidad de las líneas con un fondo blanco
plt.axis('off')
plt.title("Diagrama de Transporte TechHardware Corp.", fontsize=16, fontweight='bold')

# Mostrar el grafo
plt.show()