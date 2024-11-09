# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")


sales_rep_agent = Agent(
    role="Representante de Ventas",
    goal="Identificar leads de alto valor que coincidan "
         "con nuestro perfil de cliente ideal",
    backstory=(
        "Como parte del dinámico equipo de ventas en CrewAI, "
        "tu misión es explorar "
        "el panorama digital en busca de posibles leads. "
        "Equipado con herramientas de última generación "
        "y una mentalidad estratégica, analizas datos, "
        "tendencias e interacciones para "
        "descubrir oportunidades que otros podrían pasar por alto. "
        "Tu trabajo es crucial para allanar el camino "
        "hacia compromisos significativos y para impulsar el crecimiento de la empresa."
    ),
    allow_delegation=False,
    verbose=True
)


lead_sales_rep_agent = Agent(
    role="Representante Principal de Ventas",
    goal="Cultivar leads con comunicaciones personalizadas y atractivas",
    backstory=(
        "Dentro del vibrante ecosistema del departamento de ventas de CrewAI, "
        "te destacas como el puente entre clientes potenciales "
        "y las soluciones que necesitan."
        "Al crear mensajes atractivos y personalizados, "
        "no solo informas a los leads sobre nuestras ofertas, "
        "sino que también los haces sentir vistos y escuchados."
        "Tu papel es fundamental para convertir el interés "
        "en acción, guiando a los leads a través del viaje "
        "desde la curiosidad hasta el compromiso."
    ),
    allow_delegation=False,
    verbose=True
)


from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool

directory_read_tool = DirectoryReadTool(directory='./instructions')
file_read_tool = FileReadTool()
search_tool = SerperDevTool()


from crewai_tools import BaseTool

class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = (
        "Analiza el sentimiento de un texto "
        "para garantizar una comunicación positiva y atractiva."
    )
    
    def _run(self, text: str) -> str:
        # Tu código personalizado para la herramienta va aquí
        return "positivo"


sentiment_analysis_tool = SentimentAnalysisTool()

lead_profiling_task = Task(
    description=(
        "Realiza un análisis profundo de {lead_name}, "
        "una empresa del sector {industry} "
        "que recientemente mostró interés en nuestras soluciones. "
        "Utiliza todas las fuentes de datos disponibles "
        "para compilar un perfil detallado, "
        "enfocándote en los principales tomadores de decisiones, "
        "desarrollos empresariales recientes y necesidades potenciales "
        "que se alineen con nuestras ofertas. "
        "Esta tarea es crucial para personalizar "
        "nuestra estrategia de interacción de manera efectiva.\n"
        "No hagas suposiciones y "
        "usa solo información de la que estés completamente seguro."
    ),
    expected_output=(
        "Un informe completo sobre {lead_name}, "
        "incluyendo antecedentes de la empresa, "
        "personal clave, hitos recientes y necesidades identificadas. "
        "Destaca las áreas potenciales donde "
        "nuestras soluciones pueden aportar valor "
        "y sugiere estrategias de interacción personalizadas."
    ),
    tools=[directory_read_tool, file_read_tool, search_tool],
    agent=sales_rep_agent,
)

personalized_outreach_task = Task(
    description=(
        "Utilizando los conocimientos obtenidos del "
        "informe de perfil de {lead_name}, "
        "elabora una campaña de alcance personalizada "
        "dirigida a {key_decision_maker}, "
        "el/la {position} de {lead_name}. "
        "La campaña debe abordar su reciente {milestone} "
        "y cómo nuestras soluciones pueden apoyar sus objetivos. "
        "Tu comunicación debe resonar con la cultura y los valores "
        "de la empresa {lead_name}, "
        "demostrando un profundo entendimiento de "
        "su negocio y sus necesidades.\n"
        "No hagas suposiciones y usa solo "
        "información de la que estés completamente seguro."
    ),
    expected_output=(
        "Una serie de borradores de correos electrónicos personalizados "
        "dirigidos a {lead_name}, "
        "enfocados específicamente en {key_decision_maker}. "
        "Cada borrador debe incluir "
        "una narrativa convincente que conecte nuestras soluciones "
        "con sus logros recientes y sus objetivos futuros. "
        "Asegúrate de que el tono sea atractivo, profesional "
        "y alineado con la identidad corporativa de {lead_name}."
    ),
    tools=[sentiment_analysis_tool, search_tool],
    agent=lead_sales_rep_agent,
)


crew = Crew(
    agents=[sales_rep_agent, 
            lead_sales_rep_agent],
    
    tasks=[lead_profiling_task, 
           personalized_outreach_task],
	
    verbose=2,
	memory=True
)

inputs = {
    "lead_name": "DeepLearningAI",
    "industry": "Online Learning Platform",
    "key_decision_maker": "Andrew Ng",
    "position": "CEO",
    "milestone": "product launch"
}

result = crew.kickoff(inputs=inputs)