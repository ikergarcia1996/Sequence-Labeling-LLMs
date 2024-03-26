def get_f1(filename: str):
    try:
        with open(filename, "r") as f:
            lines = f.readlines()

            f1 = float(lines[-2].split("F1:")[-1].strip())
    except:
        # print(f"File {filename} does not contain F1 score")
        return -1
    return f1


def get_f1_model(model_name):
    results_single = []
    result_multilingual = []
    result_all = []

    for dataset in ["ncbi-disease", "bc5cdr_disease", "bc5cdr_chemical"]:
        # Single task
        filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/en/en-{dataset}/{model_name}/en-{dataset}-test_constrained.txt"
        f1 = get_f1(filename)
        results_single.append(f1)
        # multilingual
        result_multilingual.append("-")
        # all
        filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/all/{model_name}_all/en-{dataset}-test_constrained.txt"
        f1 = get_f1(filename)
        result_all.append(f1)

    for dataset in ["diann"]:
        for lang in ["en", "es"]:
            # Single task
            filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/{lang}/{lang}-{dataset}/{model_name}/{lang}-{dataset}-test_constrained.txt"
            f1 = get_f1(filename)
            results_single.append(f1)
            # multilingual
            filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/langs/{dataset}/{model_name}_lang/{lang}-{dataset}-test_constrained.txt"
            f1 = get_f1(filename)
            result_multilingual.append(f1)
            # all
            filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/all/{model_name}_all/{lang}-{dataset}-test_constrained.txt"
            f1 = get_f1(filename)
            result_all.append(f1)

        for dataset in ["e3c"]:
            for lang in ["en", "es", "fr", "it"]:
                # Single task
                filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/{lang}/{lang}-{dataset}/{model_name}/{lang}-{dataset}-test_constrained.txt"
                f1 = get_f1(filename)
                results_single.append(f1)
                # multilingual
                filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/langs/{dataset}/{model_name}_lang/{lang}-{dataset}-test_constrained.txt"
                f1 = get_f1(filename)
                result_multilingual.append(f1)
                # all
                filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/all/{model_name}_all/{lang}-{dataset}-test_constrained.txt"
                f1 = get_f1(filename)
                result_all.append(f1)

        for dataset in ["pharmaconer-bsc"]:
            for lang in ["es"]:
                # Single task
                filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/{lang}/{lang}-{dataset}/{model_name}/{lang}-{dataset}-test_constrained.txt"
                f1 = get_f1(filename)
                results_single.append(f1)
                # multilingual
                result_multilingual.append("-")
                # all
                filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/all/{model_name}_all/{lang}-{dataset}-test_constrained.txt"
                f1 = get_f1(filename)
                result_all.append(f1)

        for dataset in ["neoplasm"]:
            for lang in ["en", "es", "fr", "it"]:
                for task in ["neoplasm", "glaucoma", "mixed"]:
                    # Single task
                    filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/{lang}/{lang}-{dataset}/{model_name}/{lang}-{task}-test_constrained.txt"
                    f1 = get_f1(filename)
                    results_single.append(f1)
                    # multilingual
                    filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/langs/{dataset}/{model_name}_lang/{lang}-{task}-test_constrained.txt"
                    f1 = get_f1(filename)
                    result_multilingual.append(f1)

                    # all
                    filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_5/all/{model_name}_all/{lang}-{task}-test_constrained.txt"
                    f1 = get_f1(filename)
                    result_all.append(f1)

    assert len(results_single) == len(result_multilingual) == len(result_all)

    for i in range(len(results_single)):
        print(
            f"{results_single[i]*100 if type(results_single[i])==float else '-'}\t{result_multilingual[i]*100 if type(result_multilingual[i])==float else '-'}\t{result_all[i]*100 if type(result_all[i])==float else '-'}"
        )


def get_f1_model_zero(model_name):
    for dataset in ["e3c"]:
        for lang in ["en"]:
            for t_lang in ["en", "es", "fr", "it"]:
                # Single task
                filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_zero/{lang}/{lang}-{dataset}/{model_name}/{t_lang}-{dataset}-test_constrained.txt"
                f1 = get_f1(filename) * 100
                print(f"{f1},e3c-{t_lang} ")

    for dataset in ["neoplasm"]:
        for lang in ["en"]:
            for t_lang in ["en", "es", "fr", "it"]:
                for task in ["neoplasm", "glaucoma", "mixed"]:
                    # Single task
                    filename = f"/ikerlariak/igarcia945/Sequence Labeling LLMs/Antidote_mT5_zero/{lang}/{lang}-{dataset}/{model_name}/{t_lang}-{task}-test_constrained.txt"
                    f1 = get_f1(filename) * 100
                    print(f"{f1},{task}-{t_lang} ")


if __name__ == "__main__":
    for model_name in [
        "google_mt5-large",
        "MedicalMT5_mT5-large",
        "google_mt5-xl",
        "MedicalMT5_mT5-xl",
        "razent_SciFive-large-Pubmed_PMC",
        "google_flan-t5-large",
        "google_flan-t5-xl",
    ]:
        print(model_name)
        get_f1_model(model_name)
        print()

    print("\n\n\n")
    for model_name in [
        "google_mt5-large",
        "_gaueko1_hizkuntza-ereduak_medT5_medT5-large",
        "google_mt5-xl",
        "_gaueko1_hizkuntza-ereduak_medT5_medT5-xl",
        "razent_SciFive-large-Pubmed_PMC",
        "google_flan-t5-large",
        "google_flan-t5-xl",
    ]:
        print(model_name)
        get_f1_model_zero(model_name)
        print()
