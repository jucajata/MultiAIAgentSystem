# Control de advertencias
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

planificador = Agent(
    role="Planificador de Contenidos",
    goal="Planear contenido atractivo y fáctico sobre {tema}",
    backstory="Estás trabajando en la planificación de un artículo de blog "
              "sobre el tema: {tema}. "
              "Recolectas información que ayuda a la "
              "audiencia a aprender algo "
              "y tomar decisiones informadas. "
              "Tu trabajo es la base para que el "
              "Escritor de Contenidos redacte un artículo sobre este tema.",
    allow_delegation=False,
    verbose=True
)

escritor = Agent(
    role="Escritor de Contenidos",
    goal="Escribir un artículo de opinión perspicaz y fáctico "
         "sobre el tema: {tema}",
    backstory="Estás trabajando en la redacción "
              "de un nuevo artículo de opinión sobre el tema: {tema}. "
              "Te basas en el trabajo del Planificador de Contenidos, "
              "quien proporciona un esquema "
              "y contexto relevante sobre el tema. "
              "Sigues los objetivos y la dirección del esquema, "
              "según lo proporcionado por el Planificador de Contenidos. "
              "También ofreces ideas objetivas e imparciales "
              "y las respaldas con información "
              "proporcionada por el Planificador de Contenidos. "
              "Reconoces en tu artículo de opinión "
              "cuando tus declaraciones son opiniones "
              "en lugar de afirmaciones objetivas.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Editar una entrada de blog para alinearla con "
         "el estilo de redacción de la organización.",
    backstory="Eres un editor que recibe una entrada de blog "
              "del Escritor de Contenidos. "
              "Tu objetivo es revisar la entrada de blog "
              "para asegurarte de que siga las mejores prácticas periodísticas, "
              "proporcione puntos de vista equilibrados "
              "al expresar opiniones o afirmaciones, "
              "y evite temas o opiniones controvertidas "
              "cuando sea posible.",
    allow_delegation=False,
    verbose=True
)

planificar = Task(
    description=(
        "1. Priorizar las últimas tendencias, principales actores "
        "y noticias relevantes sobre {tema}.\n"
        "2. Identificar la audiencia objetivo, considerando "
        "sus intereses y puntos de dolor.\n"
        "3. Desarrollar un esquema de contenido detallado que incluya "
        "una introducción, puntos clave y una llamada a la acción.\n"
        "4. Incluir palabras clave SEO y datos o fuentes relevantes."
    ),
    expected_output="Un documento de planificación de contenido completo "
                    "con un esquema, análisis de audiencia, "
                    "palabras clave SEO y recursos.",
    agent=planificador,
)

escribir = Task(
    description=(
        "1. Usar el plan de contenido para crear una entrada de blog "
        "atractiva sobre {tema}.\n"
        "2. Incorporar palabras clave SEO de manera natural.\n"
        "3. Las secciones/títulos secundarios están correctamente "
        "nombrados de manera atractiva.\n"
        "4. Asegurarse de que la publicación esté estructurada "
        "con una introducción atractiva, un cuerpo informativo "
        "y una conclusión resumida.\n"
        "5. Revisar errores gramaticales y alineación con la voz de la marca.\n"
    ),
    expected_output="Una entrada de blog bien redactada "
                    "en formato markdown, lista para su publicación. "
                    "Cada sección debe tener 2 o 3 párrafos.",
    agent=escritor,
)

editar = Task(
    description=("Revisar la entrada de blog para "
                 "corrección gramatical y "
                 "alineación con la voz de la marca."),
    expected_output="Una entrada de blog bien redactada en formato markdown, "
                    "lista para su publicación. "
                    "Cada sección debe tener 2 o 3 párrafos.",
    agent=editor
)

equipo = Crew(
    agents=[planificador, escritor, editor],
    tasks=[planificar, escribir, editar],
    verbose=2
)

resultado = equipo.kickoff(inputs={"tema": "Ingeniería de datos en el sector eléctrico"})
