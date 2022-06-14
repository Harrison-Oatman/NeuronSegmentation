from .inference import inferenceloader
from . import const
from tqdm import tqdm
from .model import ffn
import matplotlib.pyplot as plt


class InferenceController:

    def __init__(self, json_file=const.INFERENCE_DIR+"inference.json"):
        self.inference_loader = inferenceloader.InferenceLoader(json_file)

    def inference(self, model: ffn.FFN):
        for batch in tqdm(self.inference_loader):
            model.start_inference_batch()

            batch.initialize_with_queue(model.movequeue)

            input_init = batch.first_pass()
            input_seeded_init = model.pom.start_batch(input_init)

            logits = model.net(input_seeded_init, training=False)
            model.apply_inference(logits)

            for input_fov, offsets in batch:
                input_pom = model.pom.request_poms(input_fov, offsets)

                logits = model.net(input_pom, training=False)
                model.apply_inference(logits, inference_step=True)

            plt.imshow(model.pom.poms[0][0])
            plt.show()


