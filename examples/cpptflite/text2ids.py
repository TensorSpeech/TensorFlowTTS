import sys

if __name__ == "__main__":
    argvs = sys.argv

    if (len(argvs) != 2):
        print("usage: python3 {} text(hanzi)".format(argvs[0]))
    else:
        from tensorflow_tts.inference import AutoProcessor
        mapper_json = "../../tensorflow_tts/processor/pretrained/baker_mapper.json"
        processor = AutoProcessor.from_pretrained(pretrained_path=mapper_json)

        input_text = argvs[1]
        input_ids = processor.text_to_sequence(input_text, inference=True)
        print(" ".join(str(i) for i in input_ids))