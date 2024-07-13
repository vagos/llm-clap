from transformers import ClapModel, ClapProcessor, AutoTokenizer
import io
import librosa
import llm


@llm.hookimpl
def register_embedding_models(register):
    register(ClapEmbeddingModel())


class ClapEmbeddingModel(llm.EmbeddingModel):
    model_id = "clap"
    supports_binary = True
    supports_text = True
    model_name = "laion/larger_clap_general"

    def __init__(self):
        self._model = None

    def embed_batch(self, items):
        # Embeds a mix of text strings and binary images
        if self._model is None:
            self._model = ClapModel.from_pretrained(self.model_name)

        embeddings = []
        processor = ClapProcessor.from_pretrained(self.model_name)
        tokenizers = AutoTokenizer.from_pretrained(self.model_name)

        for item in items:
            if isinstance(item, bytes):
                # If the item is a byte string, treat it as audio data
                audio_array, _ = librosa.load(io.BytesIO(item), sr=48000)
                if len(audio_array.shape) > 1:
                    audio_array = librosa.to_mono(audio_array)
                inputs = processor(audios=audio_array, sampling_rate=48000, return_tensors="pt")
                embedding = self._model.get_audio_features(**inputs)
            elif isinstance(item, str):
                # If the item is a string, embed it as text
                inputs = tokenizers(item, padding=True, return_tensors="pt")
                embedding = self._model.get_text_features(**inputs)
            else:
                raise ValueError(f"Cannot embed item of type: {type(item)}")

            yield [float(num) for num in embedding.flatten()]
