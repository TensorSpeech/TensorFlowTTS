import sys

if __name__ == "__main__":
    argvs = sys.argv

    if (len(argvs) != 3):
        print("usage: python3 {} mapper.json text(hanzi)".format(argvs[0]))
    else:
        from tensorflow_tts.inference import AutoProcessor
        mapper_json = argvs[1]
        processor = AutoProcessor.from_pretrained(pretrained_path=mapper_json)

        input_text = argvs[2]
        input_ids = processor.text_to_sequence(input_text, inference=True)
        print(" ".join(str(i) for i in input_ids))