from ml4vision.cli_options import Options
from ml4vision import cli_functions
import getpass
from ml4vision import __version__


def main():
    
    args, parser = Options().parse_args()
    exec_command(args, parser)

def exec_command(args, parser):

    if args.command == "authenticate":
        apikey = getpass.getpass(prompt="API key: ", stream=None)
        apikey = apikey.strip()
        if apikey == "":
            print("No API key entered. Consult the ml4vision documentation on how to get your API key.")
            return
        else:
            cli_functions.authenticate(apikey)

    if args.command == "version":
        print(__version__)

    if args.command == "project": 
        if args.action == "pull":
            cli_functions.pull_project(args.project, args.format)
        elif args.action == "push":
            cli_functions.push_to_project(args.project, args.path)

        
if __name__ == "__main__":
    main()