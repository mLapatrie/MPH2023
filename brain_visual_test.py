from visbrain.objects import BrainObj
from visbrain.scene import SceneObj

# Create a brain object
brain = BrainObj('brain', size=(500, 400), views=['lat', 'med'])

# Create a scene and add the brain to it
scene = SceneObj(size=(800, 600))
scene.add_to_subplot(brain, row=0, col=0, title="3D Brain")

# Show the scene
scene.preview()