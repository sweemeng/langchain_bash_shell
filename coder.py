import argparse
import subprocess

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Translate a task to a shell command.")
    parser.add_argument("tasks", help="Tasks to translate to shell commands.")
    # add optional argument --run for parser
    parser.add_argument("-r", "--run", default=False, action="store_true", help="Run the shell command.")

    args = parser.parse_args()
    llm = OpenAI(temperature=0)
    template = "Show me the shell command to do {task}. The command is:"
    prompt = PromptTemplate(
        input_variables=["task"],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(task=args.tasks)
    if args.run:
        subprocess.run(output, shell=True)
    else:
        print(output)


if __name__ == "__main__":
    main()

