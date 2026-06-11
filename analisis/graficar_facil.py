#importar librerias
import json
import copy
import matplotlib.pyplot as plt
import numpy as np
import sys

#función para generar gráficas
def graficar_(experimentos, resultados, guardar, ver=False):
    for i in range(len(experimentos)):
        try:
            with open(experimentos[i], "r", encoding="utf-8") as a:
                base = json.load(a)
        except:
            print("NO SE ENCONTRÓ ", experimentos[i])
            continue

        try:
            with open(resultados[i], "r", encoding="utf-8") as f:
                res = json.load(f)
        except:
            print("NO SE ENCONTRÓ ", resultados[i])
            continue

        if "Población final" in res:
            fig, axs=plt.subplots(3, 2, figsize=(10, 12))
        else:
            fig, axs=plt.subplots(2, 2, figsize=(10, 8))

        #---------------------------------------------------------
        axs[0, 0].plot([i for i in range(len(res["Best loss"]))], res["Best loss"], label=f"Mejor loss ({min(res["Best loss"])})")
        axs[0, 0].plot([i for i in range(len(res["Worst loss"]))], res["Worst loss"], label=f"Peor loss ({max(res["Worst loss"])})")
        axs[0, 0].set_xlabel("Iteraciones")
        axs[0, 0].set_ylabel("Loss ("+base["Fitness"]+")")
        axs[0, 0].set_title("Loss obtenido en\n"+resultados[i])
        axs[0, 0].grid(True)
        axs[0, 0].legend(loc="upper right")

        #---------------------------------------------------------
        axs[0, 1].plot([i for i in range(len(res["Best accuracy"]))], res["Best accuracy"], label=f"Mejor accuracy ({max(res["Best accuracy"])})")
        axs[0, 1].plot([i for i in range(len(res["Worst accuracy"]))], res["Worst accuracy"], label=f"Peor accuracy ({min(res["Worst accuracy"])})")
        axs[0, 1].set_xlabel("Iteraciones")
        axs[0, 1].set_ylabel("Accuracy")
        axs[0, 1].set_title("Accuracy obtenido en\n"+resultados[i])
        axs[0, 1].grid(True)
        axs[0, 1].legend(loc="upper right")

        #---------------------------------------------------------
        superior = np.array(res["Promedio"]) + np.array(res["Desvest"])
        inferior = np.array(res["Promedio"]) - np.array(res["Desvest"])
        axs[1, 0].plot([i for i in range(len(res["Best iteration"]))], res["Best iteration"], label=f"Mejor fitness ({min(res["Best iteration"])})")
        axs[1, 0].plot([i for i in range(len(res["Worst iteration"]))], res["Worst iteration"], label=f"Peor fitness ({max(res["Worst iteration"])})")
        axs[1, 0].plot([i for i in range(len(res["Promedio"]))], res["Promedio"], linewidth=1.5, color="teal", label="Valor promedio\npor iteracion")                # línea fuerte
        axs[1, 0].plot([i for i in range(len(superior))], superior, linestyle='--', linewidth=1.5, alpha=0.5, color="teal", label="Percentiles\n25 y 75")     # borde inferior
        axs[1, 0].plot([i for i in range(len(inferior))], inferior, linestyle='--', linewidth=1.5, alpha=0.5, color="teal")     # borde superior
        axs[1, 0].fill_between([i for i in range(len(superior))], superior, inferior, alpha=0.2, color="teal", label="±1σ")
        axs[1, 0].set_xlabel("Iteraciones")
        axs[1, 0].set_ylabel("Fitness")
        axs[1, 0].set_title("Fitness obtenido en\n"+resultados[i])
        axs[1, 0].grid(True)
        axs[1, 0].legend(loc="upper right")

        #---------------------------------------------------------
        axs[1, 1].plot([i for i in range(len(res["Succes"]))], res["Succes"], label="Succes")
        axs[1, 1].set_xlabel("Iteraciones")
        axs[1, 1].set_ylabel("Succes")
        axs[1, 1].set_title("Succes (si o no, 1 o 0)")
        axs[1, 1].grid()
        axs[1, 1].legend(loc="upper right")

        if "Población final" in res:
            axs[2, 0].boxplot(res["Población final"], 
                  notch=False,
                  tick_labels=['Población final'])

            axs[2, 0].set_title('Distribución del Fitness Final')
            axs[2, 0].set_ylabel('Fitness Final')
            axs[2, 0].grid()

            #---------------------------------------------------------
            cuantiles=np.linspace(0, 1, len(res["Población final"]))
            axs[2, 1].plot(cuantiles, res["Población final"], marker='o', label='Mi Algoritmo')
            umbral_exito=0.05
            axs[2, 1].axhline(y=umbral_exito, color='red', linestyle='--', label='Umbral de Éxito (95% de accuracy)')

            # Colorear la zona de soluciones aceptables
            axs[2, 1].fill_between(cuantiles, res["Población final"], umbral_exito, 
                            where=(np.array(res["Población final"]) <= umbral_exito), 
                            color='green', alpha=0.2, label='Zona de Éxito')
            
            axs[2, 1].set_title("QvsQ obtenido en\n"+resultados[i])
            axs[2, 1].set_xlabel('Proporción (Probabilidad Acumulada)')
            axs[2, 1].set_ylabel('Fitness de pob final')
            axs[2, 1].grid()
            axs[2, 1].legend(loc="upper right")
            

        #---------------------------------------------------------
        prom=np.mean(np.array(res["Tiempo por iteración"]))
        tot=sum(np.array(res["Tiempo por iteración"]))
        fig.suptitle(f"Resultados para "+resultados[i]+f"\n{prom:.2f} segundos por iteracion\n{tot:.2f} segundos en total\npruebas en datos de: \""+base["Reporte"]+"\"")
        plt.tight_layout()
        plt.savefig(guardar[i], dpi=300)
        if(ver==True):
            plt.show()
        else:
            plt.close()

def graficar(ruta, ver=False):
    with open(ruta, "r", encoding="utf-8") as a:
        base = json.load(a)

    experimentos=[]
    resultados=[]
    graficas=[]

    for i in range(base["N archivos"]):
        with open(base["Archivos"][i], "r", encoding="utf-8") as f:
                archvio=json.load(f)

        experimentos.append(base["Archivos"][i])
        resultados.append(archvio["Archivo salida"])
        graficas.append(archvio["Archivo graficas"])

    graficar_(experimentos, 
              resultados, 
              graficas,
              ver)

if __name__=="__main__":
    graficar(str(sys.argv[1]))