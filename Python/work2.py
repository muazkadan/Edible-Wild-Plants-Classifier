
import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow_examples.lite.model_maker.core.data_util.image_dataloader import ImageClassifierDataLoader
from tensorflow_examples.lite.model_maker.core.task import image_classifier
from tensorflow_examples.lite.model_maker.core.task.model_spec import mobilenet_v2_spec, resnet_50_spec, efficientnet_lite0_spec

train_data = ImageClassifierDataLoader.from_folder("/home/mouaz/PycharmProjects/Bitirme/dataset-jpeg")
test_data =  ImageClassifierDataLoader.from_folder("/home/mouaz/PycharmProjects/Bitirme/dataset-test-jpeg")

model = image_classifier.create(train_data, batch_size=32, model_spec=mobilenet_v2_spec, epochs=10)

loss, accuracy = model.evaluate(test_data)

model.export(export_dir='/home/mouaz/PycharmProjects/Bitirme/Android/app/src/main/assets', with_metadata=True)