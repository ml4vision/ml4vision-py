import sys
from argparse import ArgumentParser
import argcomplete

class Options:

    def __init__(self):

        self.parser = ArgumentParser(
            description="Cli for ml4vision."
        )

        subparsers = self.parser.add_subparsers(dest="command")

        # authenticate
        subparsers.add_parser("authenticate", help="Store apikey for authentication.")

        # project
        project = subparsers.add_parser("project", help="project related functions")
        project_action = project.add_subparsers(dest="action")

        ## project list
        project_list = project_action.add_parser("list", help="List all projects")
        
        ## project pull
        project_pull = project_action.add_parser("pull", help="Pull samples from project")
        project_pull.add_argument(
            "project",
            type=str,
            help="Name of project"
        )
        project_pull.add_argument(
            "--images-only",
            action="store_true",
            help="Pull the images only"
        )
        project_pull.add_argument(
            "--labels-only",
            action="store_true",
            help="Pull the labels only"
        )
        project_pull.add_argument(
            "--approved-only",
            action="store_true",
            help="Pull only the approved samples"
        )
        project_pull.add_argument(
            "--labeled-only",
            action="store_true",
            help="Pull only the labeled samples"
        )

        ## project push
        project_push = project_action.add_parser("push", help="Push images to project")
        project_push.add_argument(
            "project",
            type=str,
            help="Name of project"
        )
        project_push.add_argument(
            "path",
            type=str,
            help="Path to image folder"
        )
        project_push.add_argument(
            "--label_path",
            type=str,
            help="Path to label folder"
        )

        # version
        subparsers.add_parser("version", help="Print current version number")

        argcomplete.autocomplete(self.parser)

    def parse_args(self):
        args = self.parser.parse_args()

        if not args.command:
            self.parser.print_help()
            sys.exit()

        return args, self.parser