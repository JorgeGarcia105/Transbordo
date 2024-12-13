import pulp as op
import networkx as nx
import matplotlib.pyplot as plt

# --- DATOS DEL PROBLEMA ---

# Nodos y arcos
nodos_oferta = ["I1", "I2", "I3", "I4", "I5", "I6", "I7"]
nodos_transbordo = ["J1", "J2", "J3", "J4"]
nodos_demanda = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"]
nodos = nodos_oferta + nodos_transbordo + nodos_demanda

arcos = [
    ("I1", "J1"), ("I1", "J2"), ("I1", "K1"), ("I2", "J1"), ("I2", "J4"),
    ("I3", "J2"), ("I3", "J3"), ("I4", "J1"), ("I4", "J2"), ("I4", "J4"),
    ("I5", "J3"), ("I6", "J3"), ("I6", "J4"), ("I7", "J1"), ("I7", "J2"),
    ("I7", "K1"), ("J1", "K1"), ("J1", "K4"), ("J1", "K5"), ("J1", "K7"),
    ("J1", "K8"), ("J1", "K10"), ("J2", "K4"), ("J2", "K6"), ("J2", "K7"),
    ("J2", "K10"), ("J3", "K1"), ("J3", "K2"), ("J3", "K3"), ("J3", "K5"),
    ("J3", "K9"), ("J4", "K1"), ("J4", "K5"), ("J4", "K6")
]

# Datos de oferta, demanda y costos
oferta = {
    "I1": 300, "I2": 400, "I3": 120, "I4": 200, "I5": 380, "I6": 230, "I7": 400
}

# Aumento del 25% en la demanda
demanda = {
    "K1": 120, "K2": 100, "K3": 80, "K4": 100, "K5": 140, "K6": 200, "K7": 60,
    "K8": 90, "K9": 90, "K10": 100
}
demanda = {k: int(v * 1.25) for k, v in demanda.items()}  # Aumento de 25%

costos = {
    ("I1", "J1"): 230, ("I1", "J2"): 300, ("I1", "K1"): 290, ("I2", "J1"): 250,
    ("I2", "J4"): 280, ("I3", "J2"): 325, ("I3", "J3"): 315, ("I4", "J1"): 190,
    ("I4", "J2"): 225, ("I4", "J4"): 325, ("I5", "J3"): 305, ("I6", "J3"): 360,
    ("I6", "J4"): 310, ("I7", "J1"): 270, ("I7", "J2"): 340, ("I7", "K1"): 190,
    ("J1", "K1"): 350, ("J1", "K4"): 290, ("J1", "K5"): 310, ("J1", "K7"): 250,
    ("J1", "K8"): 320, ("J1", "K10"): 370, ("J2", "K4"): 330, ("J2", "K6"): 260,
    ("J2", "K7"): 280, ("J2", "K10"): 335, ("J3", "K1"): 370, ("J3", "K2"): 190,
    ("J3", "K3"): 305, ("J3", "K5"): 330, ("J3", "K9"): 260, ("J4", "K1"): 190,
    ("J4", "K5"): 305, ("J4", "K6"): 345
}

# --- MODELO ---

# Crear el modelo de optimización
prob = op.LpProblem("ProblemaDeTransporte", op.LpMinimize)

# Variables de decisión
x = op.LpVariable.dicts("x", arcos, lowBound=0, cat=op.LpContinuous)

# Función objetivo
prob += op.lpSum([costos[i, j] * x[i, j] for (i, j) in arcos]), "CostoTotal"

# --- RESTRICCIONES ---

# Restricciones de oferta
prob += x["I1", "J1"] + x["I1", "J2"] + x["I1", "K1"] <= 300, "Oferta_I1"
prob += x["I2", "J1"] + x["I2", "J4"] <= 400, "Oferta_I2"
prob += x["I3", "J2"] + x["I3", "J3"] <= 360, "Oferta_I3"
prob += x["I4", "J1"] + x["I4", "J2"] + x["I4", "J4"] <= 200, "Oferta_I4"
prob += x["I5", "J3"] <= 380, "Oferta_I5"
prob += x["I6", "J3"] + x["I6", "J4"] <= 230, "Oferta_I6"
prob += x["I7", "J1"] + x["I7", "J2"] + x["I7", "K1"] <= 400, "Oferta_I7"

# Restricciones de demanda
prob += x["I1", "K1"] + x["J1", "K1"] + x["J3", "K1"] + x["J4", "K1"] + x["I7", "K1"] == demanda["K1"], "Demanda_K1"
prob += x["J3", "K2"] == demanda["K2"], "Demanda_K2"
prob += x["J3", "K3"] == demanda["K3"], "Demanda_K3"
prob += x["J1", "K4"] + x["J2", "K4"] == demanda["K4"], "Demanda_K4"
prob += x["J1", "K5"] + x["J3", "K5"] + x["J4", "K5"] == demanda["K5"], "Demanda_K5"
prob += x["J2", "K6"] + x["J4", "K6"] == demanda["K6"], "Demanda_K6"
prob += x["J1", "K7"] + x["J2", "K7"] == demanda["K7"], "Demanda_K7"
prob += x["J1", "K8"] == demanda["K8"], "Demanda_K8"
prob += x["J3", "K9"] == demanda["K9"], "Demanda_K9"
prob += x["J1", "K10"] + x["J2", "K10"] == demanda["K10"], "Demanda_K10"

# Restricciones de transbordo
prob += x["I1", "J1"] + x["I2", "J1"] + x["I4", "J1"] + x["I7", "J1"] == x["J1", "K1"] + x["J1", "K4"] + x["J1", "K5"] + x["J1", "K7"] + x["J1", "K8"] + x["J1", "K10"], "Transbordo_J1"
prob += x["I1", "J2"] + x["I3", "J2"] + x["I4", "J2"] + x["I7", "J2"] == x["J2", "K4"] + x["J2", "K6"] + x["J2", "K7"] + x["J2", "K10"], "Transbordo_J2"
prob += x["I3", "J3"] + x["I5", "J3"] + x["I6", "J3"] == x["J3", "K1"] + x["J3", "K2"] + x["J3", "K3"] + x["J3", "K5"] + x["J3", "K9"], "Transbordo_J3"
prob += x["I2", "J4"] + x["I4", "J4"] + x["I6", "J4"] == x["J4", "K1"] + x["J4", "K5"] + x["J4", "K6"], "Transbordo_J4"

# --- RESOLVER EL MODELO ---
prob.solve()

# Mostrar resultados
print("Estado:", op.LpStatus[prob.status])
print("Costo total (Z):", op.value(prob.objective))
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)

# --- GRAFICAR RESULTADOS ---

# Crear grafo con NetworkX
G = nx.DiGraph()
for (i, j) in arcos:
    if x[i, j].varValue > 0:
        G.add_edge(i, j, weight=x[i, j].varValue)

# Posiciones personalizadas con mayor separación
pos = {
    "I1": (0, 0), "I2": (0, -15), "I3": (0, -25), "I4": (0, -35),
    "I5": (0, -20), "I6": (0, -35), "I7": (0, -45),  # Oferta
    "J1": (4, 0), "J2": (4, -5), "J3": (4, -30), "J4": (4, -50),  # Transbordo
    "K1": (8, 0), "K2": (8, -15), "K3": (8, -60), "K4": (8, -20),
    "K5": (8, -4), "K6": (8, -50), "K7": (8, -30), "K8": (8, -25),
    "K9": (8, -55), "K10": (8, -40),  # Demanda
}

# Dibujar el grafo
plt.figure(figsize=(15, 10))  # Ajustar tamaño del gráfico
nx.draw_networkx(G, pos, with_labels=True, node_size=3000, node_color="lightblue")
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in labels.items()})
plt.title("Flujo Óptimo")
plt.show()