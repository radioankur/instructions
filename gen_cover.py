import tempfile
import vertexai
from enum import Enum
from PIL import Image

from vertexai.preview.vision_models import ImageGenerationModel

def read_text_file(file_path):
  """
  Reads a text file and returns its content as a string.

  Args:
    file_path: The path to the text file.

  Returns:
    A string containing the file's content.
  """
  try:
    with open(file_path, 'r') as file:
      return file.read()
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return None

vertexai.init(project="stations-243022", location="us-central1")

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

TITLE = "greatest pack ever"
BACKGROUND = "cute dogs and cats"
PROMPT = read_text_file(f"gen_cover_prompt_template.txt").format(title=TITLE, background=BACKGROUND)

images = imagen_model.generate_images(
    prompt=PROMPT,
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
