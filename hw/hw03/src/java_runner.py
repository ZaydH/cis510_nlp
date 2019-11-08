from pathlib import Path
import subprocess
from typing import Union

JAVA_DIR = Path("maxent")
MAX_ENT_JAR = JAVA_DIR / "maxent-3.0.0.jar"
TROVE_JAR = JAVA_DIR / "trove.jar"

ME_TRAIN_PATH = JAVA_DIR / "MEtrain.java"
ME_TAG_PATH = JAVA_DIR / "MEtag.java"


def _compile_java_file(file: Union[Path, str]) -> None:
    r""" Compiles the specified Java file """
    file = Path(file)
    if not file.exists(): raise ValueError(f"Unable to find file to compile: {file}")
    if file.suffix != ".java": raise ValueError(f"{file} does not appear to be a java file")

    cmd = f"javac -cp \"{JAVA_DIR}:{MAX_ENT_JAR}\" {file}"
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()


def run_java_class(class_name: str, *args) -> None:
    r"""
    Runs the specified Java class.

    :param class_name: Name of the Java class
    :param args: Arguments to pass to \p class_name's \p main method
    """
    args = " ".join(args) if len(args) > 0 else ""
    cmd = f"javac -cp \"{MAX_ENT_JAR}:{TROVE_JAR}:{JAVA_DIR}\" {class_name} {args}"
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()


def train_maxent_model(data_path: Path, model_path: Path) -> None:
    r"""
    Train max entropy model

    :param data_path: Path to the data file
    :param model_path: Path where to save the trained model
    """
    _compile_java_file(ME_TRAIN_PATH)
    if not model_path.parent.exists(): model_path.parent.mkdir(exist_ok=True, parents=True)

    class_name = ME_TRAIN_PATH.stem
    run_java_class(class_name, data_path, model_path)


def label_with_maxent(data_path: Path, model_path: Path, output_file: Path) -> None:
    r"""
    Label the tokens in the \p data_path file using the model in \p model_path and write it to
    file \p output_file.

    :param data_path: Path to the data file
    :param model_path: Path to the previously training model
    :param output_file: Location to write labeled output
    """
    _compile_java_file(ME_TAG_PATH)
    if not model_path.exists(): raise ValueError(f"Model path file \"{model_path}\" does not exist")

    class_name = ME_TAG_PATH.stem
    run_java_class(class_name, data_path, model_path, output_file)
