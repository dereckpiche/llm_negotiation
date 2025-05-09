def output_source_code(model, output_path: str) -> None:
    """
    Outputs the source code of the model to the given path.
    """
    with open(output_path, "w") as f:
        f.write(model.source_code)
