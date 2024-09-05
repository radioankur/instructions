import tempfile
import vertexai
from PIL import Image

from vertexai.preview.vision_models import ImageGenerationModel

vertexai.init(project="stations-243022", location="us-central1")

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

# Update me.
PROMPT = "A cute dog"

images = imagen_model.generate_images(
    prompt=PROMPT,
    number_of_images=6,
    aspect_ratio="9:16",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

for image in images:
    tmp = tempfile.NamedTemporaryFile(dir='/tmp')
    image.save(location=tmp.name, include_generation_parameters=False)
    img = Image.open(tmp.name)
    img.show()