import tempfile
import vertexai
from enum import Enum
from PIL import Image

class Style(Enum):
    CINEMATIC = "cinematic"
    ANIME = "anime"
    PIXEL_ART = "pixel art"
    FANTASY = "fantasy"
    # TODO add more.

from vertexai.preview.vision_models import ImageGenerationModel

vertexai.init(project="stations-243022", location="us-central1")

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

SYSTEM_INSTRUCTION = """You are a helpful and creative thumbnail designer. 
Create an image of PROMPT with the \"TITLE\" written in a creative font across the center that is bold and eye-catching.
Make sure the words are human-readable and displayed exactly as written. 

# Update me.
TITLE = "Pawsome Pals"
PROMPT = "A cute dogs and cats"
PROMPT_TEMPLATE = "Display '{}' in a bold and eye-catching font. Make the background {}"

images = imagen_model.generate_images(
    prompt=PROMPT_TEMPLATE.format(TITLE, PROMPT),
    number_of_images=4,
    aspect_ratio="3:4",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

for image in images:
    tmp = tempfile.NamedTemporaryFile(dir='/tmp')
    image.save(location=tmp.name, include_generation_parameters=False)
    img = Image.open(tmp.name)
    img.show()
