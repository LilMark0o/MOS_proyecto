import math
import os
import time

import folium
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
import requests as req

# ------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------------------------

API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
if API_KEY is None:
    print("Error: La variable de entorno 'GOOGLE_MAPS_API_KEY' no está configurada.")
    exit()


def get_distance(lat1, lon1, lat2, lon2):
    res = req.post(
        'https://routes.googleapis.com/directions/v2:computeRoutes',
        headers={
            'Content-Type': 'application/json',
            'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters',
            'X-Goog-Api-Key': API_KEY
        },
        json={
            "origin": {
                "location": {
                    "latLng": {
                        "latitude": lat1,
                        "longitude": lon1
                    }
                }
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": lat2,
                        "longitude": lon2
                    }
                }
            },
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_UNAWARE",
            "computeAlternativeRoutes": False,
            "routeModifiers": {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": False
            },
            "languageCode": "en-US",
            "units": "IMPERIAL"
        }
    )
    return res.json()['routes'][0]['distanceMeters'] / 1000  # Convertir a km


# ----------------------------------------------------------
# DATOS CSV
# ----------------------------------------------------------
# Leer datos desde CSV
depots_df = pd.read_csv("depots.csv")
clients_df = pd.read_csv("clients.csv")
vehicles_df = pd.read_csv("vehicles.csv")

# Nombres de nodos
depots = [f"CD{int(i)}" for i in depots_df["DepotID"]]
clients = [f"C{int(i)}" for i in clients_df["ClientID"]]
nodes = depots + clients
vehicles = [f"V{int(i)}" for i in vehicles_df["VehicleID"]]

# Coordenadas de todos los nodos
coords = {
    **{f"CD{int(row.DepotID)}": (row.Latitude, row.Longitude) for _, row in depots_df.iterrows()},
    **{f"C{int(row.ClientID)}": (row.Latitude, row.Longitude) for _, row in clients_df.iterrows()}
}

# Demanda (0 para depósitos)
demand = {f"C{int(row.ClientID)}": row.Demand for _,
          row in clients_df.iterrows()}
demand.update({d: 0 for d in depots})

# Stock en depósitos
stock = {f"CD{int(row.DepotID)}": row.Capacity for _,
         row in depots_df.iterrows()}

# Capacidad y rango de vehículos
vehicle_capacity = {
    f"V{int(row.VehicleID)}": row.Capacity for _, row in vehicles_df.iterrows()}
vehicle_range = {f"V{int(row.VehicleID)}": row.Range for _,
                 row in vehicles_df.iterrows()}


# 5. MATRIZ DE DISTANCIAS
# ------------------------------------------------------------------------------
# Cálculo de distancias entre todos los pares de nodos usando la API de Google Maps
# Esta matriz es fundamental para determinar los costos de transporte y verificar
# restricciones de rango.

# Calcular la distancia entre cada par de nodos utilizando la API de Google Maps
print("Calculando matriz de distancias entre nodos...")
dist = {}
for i in nodes:
    for j in nodes:
        if i != j:
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            dist[(i, j)] = get_distance(lat1, lon1, lat2, lon2)
        else:
            dist[(i, j)] = 0  # La distancia de un nodo a sí mismo es 0

# Imprimir matriz de distancias para verificación
print("\nMatriz de distancias (km):")
print("          ", end="")
for j in nodes:
    print(f"{j:8}", end="")
print()
for i in nodes:
    print(f"{i:10}", end="")
    for j in nodes:
        print(f"{dist[(i, j)]:.2f}    ", end="")
    print()

# ------------------------------------------------------------------------------
# 6. CONSTRUCCIÓN DEL MODELO EN PYOMO
# ------------------------------------------------------------------------------
# Inicialización del modelo de optimización y definición de los conjuntos base.

print("\nCreando modelo de optimización...")
model = pyo.ConcreteModel(name="LogistiCo_Urban_Logistics")

# Definir conjuntos
# Nodos totales (centros y clientes)
model.N = pyo.Set(initialize=nodes)
model.D = pyo.Set(initialize=depots)
model.C = pyo.Set(initialize=clients)
model.V = pyo.Set(initialize=vehicles)

model.A = pyo.Set(initialize=[(i, j) for i in nodes for j in nodes if i != j])

# ------------------------------------------------------------------------------
# 7. DEFINICIÓN DE PARÁMETROS EN EL MODELO
# ------------------------------------------------------------------------------
# Configuración de todos los parámetros que el modelo utilizará para la optimización.

# Distancias entre nodos
model.dist = pyo.Param(model.A,
                       initialize=lambda model, i, j: dist[(i, j)],
                       within=pyo.NonNegativeReals,
                       doc="Distancia en km entre los nodos i y j")

# Demanda de cada nodo
model.demand = pyo.Param(model.N,
                         initialize=lambda model, i: demand[i],
                         doc="Demanda en unidades para cada nodo")

# Stock disponible en cada centro de distribución
model.stock = pyo.Param(model.D,
                        initialize=lambda model, i: stock[i],
                        doc="Inventario disponible en cada centro de distribución")

# Capacidad de cada vehículo
model.cap_vehicle = pyo.Param(model.V,
                              initialize=lambda model, v: vehicle_capacity[v],
                              doc="Capacidad de carga en unidades de cada vehículo")

# Rango máximo de operación de cada vehículo
model.range_vehicle = pyo.Param(model.V,
                                initialize=lambda model, v: vehicle_range[v],
                                doc="Rango operativo en km de cada vehículo")

# Factor de costo por km recorrido
F_t = 5000
C_m = 700
Pf = 16259

# Consumo de combustible: 0.1 litros por km
# Este valor es un promedio para vehículos de carga ligera en entorno urbano
fuel_consumption = 0.0233  # litros por km
fuel_cost_per_km = Pf * fuel_consumption

cost_factor = F_t + C_m + fuel_cost_per_km
# (lo borramos sin culpa en el ultimo commit, en el penultimo estaba pero lo borramos sin culpa)
model.cost_factor = cost_factor

# ------------------------------------------------------------------------------
# 8. VARIABLES DE DECISIÓN
# ------------------------------------------------------------------------------
# Definición de las variables que el modelo determinará para obtener la solución óptima.

# x[v,i,j] = 1 si el vehículo v viaja directamente del nodo i al nodo j, 0 en caso contrario
model.x = pyo.Var(model.V, model.A, domain=pyo.Binary,
                  doc="Variable binaria que indica si el vehículo v recorre el arco (i,j)")

# f[v,i,j]: cantidad de inventario transportado por el vehículo v en el arco (i,j)
# Esta variable ayuda a modelar el flujo de inventario a través de la red
model.f = pyo.Var(model.V, model.A, domain=pyo.NonNegativeReals,
                  doc="Flujo de inventario (unidades) transportado por el vehículo v en el arco (i,j)")

# u[v,i]: variable auxiliar para eliminar subtours (posición del nodo i en la ruta del vehículo v)
model.u = pyo.Var(model.V, model.N, domain=pyo.NonNegativeReals, bounds=(1, len(nodes)),
                  doc="Variable para eliminación de subtours - Posición del nodo en la ruta")

# Variable para identificar el depósito de inicio y fin para cada vehículo
model.start_depot = pyo.Var(model.V, model.D, domain=pyo.Binary,
                            doc="Si el vehículo v inicia su ruta en el depósito d")
model.end_depot = pyo.Var(model.V, model.D, domain=pyo.Binary,
                          doc="Si el vehículo v termina su ruta en el depósito d")

# ------------------------------------------------------------------------------
# 9. FUNCIÓN OBJETIVO
# ------------------------------------------------------------------------------
# Definición del objetivo de optimización: minimizar el costo total de transporte.

# Minimizar el costo total de los recorridos (distancia * costo_factor para cada arco utilizado)


def obj_rule(model):
    return sum(model.dist[i, j] * model.cost_factor * model.x[v, i, j]
               for v in model.V for (i, j) in model.A)


# Registrar la función objetivo en el modelo
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize,
                          doc="Minimizar el costo total de transporte")

# ------------------------------------------------------------------------------
# 10. RESTRICCIONES
# ------------------------------------------------------------------------------
# Conjunto de restricciones que definen el comportamiento y las limitaciones del sistema.

# 10.1 Cada cliente debe ser visitado exactamente una vez (por algún vehículo)
# Esta restricción garantiza que todos los clientes reciban su mercancía solo UNA vez


def cliente_unico_rule(model, j):
    if j in model.C:
        return sum(model.x[v, i, j] for v in model.V for i in model.N if i != j) == 1
    else:
        return pyo.Constraint.Skip


model.cliente_unico = pyo.Constraint(model.N, rule=cliente_unico_rule,
                                     doc="Cada cliente es visitado exactamente una vez")

# 10.2 Flujo de inventario en cada nodo:
# Modificamos la restricción original para asegurar el balance de flujo


def flow_balance_rule(model, v, j):
    if j in model.C:
        # Para clientes: flujo que entra - flujo que sale = demanda
        return (sum(model.f[v, i, j] for i in model.N if i != j) -
                sum(model.f[v, j, k] for k in model.N if k != j)) == model.demand[j] * sum(model.x[v, i, j] for i in model.N if i != j)
    else:
        # Para depósitos: permitimos más salida que entrada
        return pyo.Constraint.Skip


model.flow_balance = pyo.Constraint(model.V, model.N, rule=flow_balance_rule,
                                    doc="Balance de flujo de inventario en nodos")

# 10.3 Vinculación entre la ruta y el flujo:
# El flujo en el arco (i,j) solo puede ser positivo si el vehículo recorre ese arco


def capacity_link_rule(model, v, i, j):
    return model.f[v, i, j] <= model.cap_vehicle[v] * model.x[v, i, j]


model.capacity_link = pyo.Constraint(model.V, model.A, rule=capacity_link_rule,
                                     doc="Vinculación entre rutas y flujo de inventario")

# 10.4 Restricción de rango operativo para cada vehículo
# La distancia total recorrida por cada vehículo no debe exceder su rango máximo


def range_rule(model, v):
    return sum(model.dist[i, j] * model.x[v, i, j] for (i, j) in model.A) <= model.range_vehicle[v]


model.range_constr = pyo.Constraint(model.V, rule=range_rule,
                                    doc="Restricción de rango operativo de los vehículos")

# 10.5 Restricción de stock en cada centro de distribución:
# La suma total de inventario despachado desde un centro no debe superar su stock


def depot_stock_rule(model, d):
    return sum(model.f[v, d, j] for v in model.V for j in model.N if d != j) <= model.stock[d]


model.depot_stock = pyo.Constraint(model.D, rule=depot_stock_rule,
                                   doc="Limitación de stock en centros de distribución")

# 10.6 Cada vehículo debe partir de exactamente un depósito


def unique_start_depot_rule(model, v):
    return sum(model.start_depot[v, d] for d in model.D) <= 1


model.unique_start = pyo.Constraint(model.V, rule=unique_start_depot_rule,
                                    doc="Cada vehículo parte de a lo sumo un depósito")

# 10.7 Cada vehículo debe terminar en exactamente un depósito


def unique_end_depot_rule(model, v):
    return sum(model.end_depot[v, d] for d in model.D) <= 1


model.unique_end = pyo.Constraint(model.V, rule=unique_end_depot_rule,
                                  doc="Cada vehículo termina en a lo sumo un depósito")

# 10.8 Vincular las variables de inicio de depósito con las variables de ruta


def start_depot_link_rule(model, v, d):
    return sum(model.x[v, d, j] for j in model.N if d != j) == model.start_depot[v, d]


model.start_depot_link = pyo.Constraint(model.V, model.D, rule=start_depot_link_rule,
                                        doc="Vinculación entre inicio de ruta y depósito seleccionado")

# 10.9 Vincular las variables de fin de depósito con las variables de ruta


def end_depot_link_rule(model, v, d):
    return sum(model.x[v, i, d] for i in model.N if i != d) == model.end_depot[v, d]


model.end_depot_link = pyo.Constraint(model.V, model.D, rule=end_depot_link_rule,
                                      doc="Vinculación entre fin de ruta y depósito seleccionado")

# 10.10 Restricción de conservación de flujo (para garantizar rutas válidas)
# Si un vehículo entra a un nodo, debe salir de él (excepto en los depósitos iniciales/finales)


def flow_conservation_rule(model, v, j):
    # Para cada vehículo v y nodo j (excepto los depósitos)
    if j in model.C:  # Solo aplicamos a clientes
        # Entradas = Salidas
        return sum(model.x[v, i, j] for i in model.N if i != j) == sum(model.x[v, j, k] for k in model.N if k != j)
    return pyo.Constraint.Skip


model.flow_conservation = pyo.Constraint(model.V, model.N, rule=flow_conservation_rule,
                                         doc="Conservación de flujo de vehículos")

# 10.11 Eliminación de subtours usando la formulación MTZ (Miller-Tucker-Zemlin)


def subtour_elimination_rule(model, v, i, j):
    if i != j and i in model.C and j in model.C:  # Solo para pares de clientes distintos
        M = (len(nodes)*2)**2  # Valor grande
        return model.u[v, j] >= model.u[v, i] + 1 - M * (1 - model.x[v, i, j])
    return pyo.Constraint.Skip


model.subtour_elimination = pyo.Constraint(model.V, model.N, model.N, rule=subtour_elimination_rule,
                                           doc="Eliminación de subtours (MTZ)")

# 10.12 Vinculación entre uso de vehículos y demanda


def vehicle_usage_rule(model, v):
    # La carga total transportada por el vehículo v debe ser <= capacidad si se usa
    total_load = sum(model.demand[c] * sum(model.x[v, i, c]
                     for i in model.N if i != c) for c in model.C)

    # Si un vehículo transporta carga, debe salir de algún depósito
    vehicle_used = sum(model.start_depot[v, d] for d in model.D)

    return total_load <= model.cap_vehicle[v] * vehicle_used


model.vehicle_usage = pyo.Constraint(model.V, rule=vehicle_usage_rule,
                                     doc="Vinculación entre uso de vehículos y demanda")

# ------------------------------------------------------------------------------
# 11. RESOLUCIÓN DEL MODELO
# ------------------------------------------------------------------------------
# Configuración del solver y ejecución de la optimización.


# Usar el solver GLPK
solver = pyo.SolverFactory('glpk')
results = solver.solve(model)

if (results.solver.status == pyo.SolverStatus.ok and
        results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("\nSolución óptima encontrada!")
elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    print("\nEl modelo es infactible. Revise las restricciones y parámetros.")
    exit()
else:
    print(
        f"\nEl solver terminó con estado: {results.solver.status}, condición: {results.solver.termination_condition}")
    print("La solución puede no ser óptima o puede haber errores.")

# ------------------------------------------------------------------------------
# 12. IMPRESIÓN Y ANÁLISIS DE RESULTADOS
# ------------------------------------------------------------------------------
# Extracción y presentación estructurada de la solución obtenida.

print("\n==== RESULTADOS DE LA OPTIMIZACIÓN ====")
print(f"Valor objetivo (Costo total): {pyo.value(model.obj):.2f} COP")

total_distance = 0
total_load = 0

print("\n==== RUTAS Y FLUJOS ÓPTIMOS POR VEHÍCULO ====")
for v in model.V:
    vehicle_distance = 0
    vehicle_load = 0
    route_nodes = []

    start_depot = None
    for d in depots:
        try:
            if pyo.value(model.start_depot[v, d]) > 0.5:
                start_depot = d
                break
        except:
            pass

    if start_depot is None:
        print(f"\nVehículo {v}:")
        print("----------------------------------------")
        print("  Este vehículo no es utilizado en la solución óptima.")
        continue

    end_depot = None
    for d in depots:
        try:
            if pyo.value(model.end_depot[v, d]) > 0.5:
                end_depot = d
                break
        except:
            pass

    print(
        f"\nVehículo {v} (depósito inicial: {start_depot}, depósito final: {end_depot}):")
    print("----------------------------------------")

    current_node = start_depot
    route_nodes.append(current_node)

    while True:
        next_node = None
        for j in nodes:
            if j != current_node:
                try:
                    x_val = pyo.value(model.x[v, current_node, j])
                    if x_val > 0.5:
                        next_node = j
                        try:
                            flow_val = pyo.value(model.f[v, current_node, j])
                            vehicle_load = max(vehicle_load, flow_val)
                            vehicle_distance += dist[(current_node, j)]
                            print(
                                f"  {current_node} -> {j} | Distancia: {dist[(current_node, j)]:.2f} km | Flujo: {flow_val:.1f} unidades")
                        except Exception as e:
                            print(
                                f"  Error evaluando flujo {v},{current_node},{j}: {e}")
                        break
                except Exception as e:
                    print(
                        f"  Error evaluando ruta {v},{current_node},{j}: {e}")

        if next_node is None:
            break

        current_node = next_node
        route_nodes.append(current_node)

        if current_node in depots and current_node == end_depot:
            break

    # Imprimir resumen del vehículo
    print(f"  Ruta completa: {' -> '.join(route_nodes)}")
    print(f"  Distancia total recorrida: {vehicle_distance:.2f} km")
    print(f"  Carga máxima transportada: {vehicle_load:.1f} unidades")
    print(
        f"  Capacidad utilizada: {(vehicle_load/vehicle_capacity[v])*100:.1f}%")
    print(f"  Costo del recorrido: {vehicle_distance * cost_factor:.2f} COP")

    total_distance += vehicle_distance
    total_load += sum(demand[n] for n in route_nodes if n in clients)

print("\n==== RESUMEN GLOBAL DE LA SOLUCIÓN ====")
print(
    f"Distancia total recorrida por todos los vehículos: {total_distance:.2f} km")
print(f"Demanda total satisfecha: {sum(demand.values()):.1f} unidades")
print(f"Costo total de operación: {total_distance * cost_factor:.2f} COP")
print(
    f"Costo promedio por unidad entregada: {(total_distance * cost_factor)/sum(demand.values()):.2f} COP/unidad")

print("\n==== ESTADÍSTICAS DE UTILIZACIÓN ====")
print("Centros de distribución:")
for d in depots:
    stock_usado = sum(pyo.value(
        model.f[v, d, j]) for v in model.V for j in nodes if j != d and (v, d, j) in model.f)
    print(
        f"  {d}: {stock_usado:.1f}/{stock[d]} unidades ({(stock_usado/stock[d])*100:.2f}% utilizado)")

print("\nVehículos:")
for v in model.V:
    dist_recorrida = sum(dist[(i, j)] * pyo.value(model.x[v, i, j])
                         for (i, j) in model.A if (v, i, j) in model.x)
    print(
        f"  {v}: {dist_recorrida:.2f}/{vehicle_range[v]} km ({(dist_recorrida/vehicle_range[v])*100:.2f}% del rango utilizado)")

# ------------------------------------------------------------------------------
# 13. EXPORTACIÓN DE RESULTADOS (OPCIONAL)
# ------------------------------------------------------------------------------
# Exportar resultados a CSV o Excel para análisis posteriores

try:
    # Crear dataframe para rutas
    routes_data = []
    for v in model.V:
        for (i, j) in model.A:
            try:
                x_val = pyo.value(model.x[v, i, j])
                if x_val > 0.5:
                    try:
                        flow_val = pyo.value(model.f[v, i, j])
                    except:
                        flow_val = 0

                    routes_data.append({
                        'vehiculo': v,
                        'origen': i,
                        'destino': j,
                        'distancia_km': dist[(i, j)],
                        'flujo_unidades': flow_val,
                        'costo': dist[(i, j)] * cost_factor
                    })
            except:
                pass

    if routes_data:
        routes_df = pd.DataFrame(routes_data)
        routes_df.to_csv('resultados_rutas_logistico.csv', index=False)
        print("\nResultados exportados a 'resultados_rutas_logistico.csv'")
except:
    print("\nNo se pudieron exportar los resultados a CSV (requiere pandas)")

print("\n==== FIN DEL PROGRAMA ====")


def plot_routes_on_map():
    all_lats = [coords[node][0] for node in nodes]
    all_lons = [coords[node][1] for node in nodes]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    for node in nodes:
        lat, lon = coords[node]
        if node in depots:
            folium.Marker(
                location=[lat, lon],
                popup=f"Centro: {node}",
                icon=folium.Icon(color='black', icon='home', prefix='fa')
            ).add_to(m)
        else:
            folium.Marker(
                location=[lat, lon],
                popup=f"Cliente: {node}",
                icon=folium.Icon(color='orange', icon='user', prefix='fa')
            ).add_to(m)

    vehicle_colors = {'V1': 'red', 'V2': 'blue', 'V3': 'green'}

    for v in vehicles:
        route_nodes = []
        start_depot = None
        for d in depots:
            try:
                if pyo.value(model.start_depot[v, d]) > 0.5:
                    start_depot = d
                    break
            except:
                pass
        if start_depot is None:
            continue

        route_nodes.append(start_depot)
        current_node = start_depot

        while True:
            next_node = None
            for j in nodes:
                if j != current_node:
                    try:
                        if pyo.value(model.x[v, current_node, j]) > 0.5:
                            next_node = j
                            break
                    except:
                        pass
            if next_node is None:
                break
            route_nodes.append(next_node)
            current_node = next_node
            try:
                if next_node in depots and pyo.value(model.end_depot[v, next_node]) > 0.5:
                    break
            except:
                pass

        route_coords = [coords[node] for node in route_nodes]
        folium.PolyLine(
            locations=[(lat, lon) for (lat, lon) in route_coords],
            color=vehicle_colors[v],
            weight=5,
            opacity=0.7,
            tooltip=f"Ruta {v}"
        ).add_to(m)
    m.save("mapa_rutas.html")


plot_routes_on_map()
