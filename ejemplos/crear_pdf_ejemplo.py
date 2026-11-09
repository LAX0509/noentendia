"""Genera ejemplos/apuntes_ia.pdf: apuntes de 5 páginas para la demo y las pruebas.

Uso:  python ejemplos/crear_pdf_ejemplo.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGINAS = [
    ("Sistemas expertos y Prolog", [
        "Un sistema experto es un programa que imita el razonamiento de un especialista "
        "humano en un área concreta, como el diagnóstico médico o la asesoría legal.",
        "Está formado por una base de conocimiento, que guarda hechos y reglas, y un motor "
        "de inferencia, que aplica esas reglas para llegar a conclusiones nuevas.",
        "Prolog es un lenguaje de programación lógica muy usado para construir sistemas "
        "expertos. En Prolog se declaran hechos, como padre(juan, ana), y reglas, como "
        "abuelo(X, Z) :- padre(X, Y), padre(Y, Z).",
        "El motor de Prolog responde consultas mediante encadenamiento hacia atrás: parte "
        "de la meta y busca reglas y hechos que la demuestren, retrocediendo cuando un "
        "camino falla. A este retroceso se le llama backtracking.",
        "Ejemplo: un sistema experto de becas puede tener la regla 'si el estudiante tiene "
        "promedio mayor a 4.0 y estrato 1 o 2, entonces es elegible'.",
    ]),
    ("Ontologías y Protégé", [
        "Una ontología es una representación formal de los conceptos de un dominio y de las "
        "relaciones entre ellos. Permite que personas y programas compartan un mismo "
        "vocabulario.",
        "Sus elementos principales son las clases, que agrupan conceptos; las instancias o "
        "individuos, que son casos concretos; y las propiedades, que relacionan clases o "
        "asignan atributos.",
        "Protégé es una herramienta gratuita de la Universidad de Stanford para crear "
        "ontologías en el lenguaje OWL. Incluye razonadores, como HermiT, que detectan "
        "inconsistencias y deducen relaciones implícitas.",
        "Ejemplo: en una ontología universitaria, Profesor y Estudiante son subclases de "
        "Persona, y la propiedad dicta relaciona a un Profesor con un Curso.",
    ]),
    ("Aprendizaje supervisado y clasificación", [
        "En el aprendizaje supervisado el modelo aprende a partir de datos etiquetados, es "
        "decir, ejemplos donde ya se conoce la respuesta correcta.",
        "La clasificación predice una categoría, por ejemplo si un correo es spam o no. "
        "Algoritmos comunes son los árboles de decisión, Random Forest, la regresión "
        "logística y las máquinas de soporte vectorial.",
        "Para validar un clasificador se separan los datos en entrenamiento y prueba. La "
        "matriz de confusión cuenta verdaderos positivos, falsos positivos, verdaderos "
        "negativos y falsos negativos.",
        "La precisión mide cuántas predicciones positivas fueron correctas y la "
        "exhaustividad o recall mide cuántos positivos reales encontró el modelo. Con datos "
        "desbalanceados, la exactitud sola puede engañar.",
    ]),
    ("Agrupamiento no supervisado", [
        "El agrupamiento o clustering es una técnica de aprendizaje no supervisado: los "
        "datos no tienen etiquetas y el algoritmo descubre grupos de elementos parecidos.",
        "K-Means divide los datos en K grupos. Elige K centroides, asigna cada punto al "
        "centroide más cercano y recalcula los centroides hasta que dejan de moverse.",
        "Como K-Means necesita conocer K de antemano, se prueba con varios valores y se "
        "compara el coeficiente de silueta, que va de -1 a 1: valores altos indican grupos "
        "compactos y bien separados.",
        "El costo de máquina de K-Means crece con el número de datos, de grupos y de "
        "iteraciones, por lo que suele ser rápido incluso con miles de registros.",
    ]),
    ("Sistemas multiagente y NetLogo", [
        "Un agente es una entidad que percibe su entorno y actúa sobre él para cumplir "
        "objetivos. Un sistema multiagente reúne varios agentes que interactúan, cooperan "
        "o compiten.",
        "NetLogo es un entorno de simulación donde las tortugas son los agentes móviles, los "
        "parches son las celdas del mundo y el observador da las órdenes globales.",
        "En una simulación de patrullaje, cada agente sigue reglas locales simples, como "
        "moverse hacia la zona menos visitada, y del conjunto surge un comportamiento "
        "global llamado comportamiento emergente.",
        "Las simulaciones permiten probar políticas, como la distribución de recursos en "
        "una emergencia, antes de aplicarlas en el mundo real.",
    ]),
]


def crear(destino: Path) -> Path:
    estilos = getSampleStyleSheet()
    historia = []
    for i, (titulo, parrafos) in enumerate(PAGINAS):
        historia.append(Paragraph(f"Tema {i + 1}. {titulo}", estilos["Heading1"]))
        for p in parrafos:
            historia += [Paragraph(p, estilos["BodyText"]), Spacer(1, 8)]
        if i < len(PAGINAS) - 1:
            historia.append(PageBreak())
    destino.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(destino), pagesize=letter,
                      title="Apuntes de Inteligencia Artificial").build(historia)
    return destino


if __name__ == "__main__":
    print(crear(Path(__file__).with_name("apuntes_ia.pdf")))
