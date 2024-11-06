# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Role Playing, Focus and Cooperation
support_agent = Agent(
    role="Representante Senior de Soporte",
    goal="Ser el representante de soporte más amigable y útil de tu equipo",
    backstory=(
        "Trabajas en crewAI (https://crewai.com) y "
        "ahora te dedicas a brindar soporte a {customer}, "
        "un cliente muy importante para tu empresa."
        "Debes asegurarte de brindar el mejor soporte posible. "
        "Proporciona respuestas completas y detalladas, "
        "y evita hacer suposiciones."
    ),
    allow_delegation=False,
    verbose=True
)

support_quality_assurance_agent = Agent(
    role="Especialista en Garantía de Calidad de Soporte",
    goal="Obtener reconocimiento por proporcionar la mejor garantía de calidad de soporte en tu equipo",
    backstory=(
        "Trabajas en crewAI (https://crewai.com) y "
        "ahora trabajas con tu equipo "
        "en una solicitud de {customer} para asegurar que "
        "el representante de soporte esté "
        "brindando el mejor soporte posible.\n"
        "Debes asegurarte de que el representante de soporte "
        "proporcione respuestas completas y detalladas, sin hacer suposiciones."
    ),
    verbose=True
)


from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool


docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
)

inquiry_resolution = Task(
    description=(
        "{customer} acaba de contactarse con una solicitud muy importante:\n"
        "{inquiry}\n\n"
        "{person} de {customer} fue quien se puso en contacto. "
        "Asegúrate de utilizar todo lo que sabes "
        "para brindar el mejor soporte posible. "
        "Debes esforzarte en proporcionar una respuesta completa "
        "y precisa a la consulta del cliente."
    ),
    expected_output=(
        "Una respuesta detallada e informativa a la "
        "consulta del cliente que aborde "
        "todos los aspectos de su pregunta.\n"
        "La respuesta debe incluir referencias "
        "a todo lo que utilizaste para encontrar la respuesta, "
        "incluyendo datos externos o soluciones. "
        "Asegúrate de que la respuesta esté completa, "
        "sin dejar preguntas sin responder, y mantén un tono "
        "amigable y servicial en todo momento."
    ),
    tools=[docs_scrape_tool],
    agent=support_agent,
)


quality_assurance_review = Task(
    description=(
        "Revisa la respuesta redactada por el Representante Senior de Soporte para la consulta de {customer}. "
        "Asegúrate de que la respuesta sea completa, precisa y cumpla con los "
        "estándares de alta calidad esperados para el soporte al cliente.\n"
        "Verifica que todas las partes de la consulta del cliente "
        "hayan sido abordadas "
        "de manera exhaustiva, con un tono amigable y servicial.\n"
        "Revisa las referencias y fuentes utilizadas para "
        "encontrar la información, "
        "asegurándote de que la respuesta esté bien fundamentada y "
        "no deje preguntas sin responder."
    ),
    expected_output=(
        "Una respuesta final, detallada e informativa "
        "lista para ser enviada al cliente.\n"
        "Esta respuesta debe abordar completamente la "
        "consulta del cliente, incorporando todos los "
        "comentarios y mejoras relevantes.\n"
        "No seas demasiado formal, somos una empresa relajada y cool, "
        "pero mantén un tono profesional y amigable en todo momento."
    ),
    agent=support_quality_assurance_agent,
)


crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution, quality_assurance_review],
  verbose=2,
  memory=True
)

inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": "Necesito ayuda para configurar un Crew "
               "y ponerlo en marcha, específicamente "
               "¿cómo puedo agregar memoria a mi crew? "
               "¿Puedes proporcionar orientación?"
}
result = crew.kickoff(inputs=inputs)
