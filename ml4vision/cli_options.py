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

        # dataset
        dataset = subparsers.add_parser("dataset", help="dataset related functions")
        dataset_action = dataset.add_subparsers(dest="action")

        ## dataset list
        dataset_list = dataset_action.add_parser("list", help="List all datasets")
        
        ## dataset pull
        dataset_pull = dataset_action.add_parser("pull", help="Pull samples from dataset")
        dataset_pull.add_argument(
            "dataset",
            type=str,
            help="Name of dataset"
        )
        dataset_pull.add_argument(
            "--approved-only",
            action="store_true",
            help="Pull only the approved samples"
        )
        dataset_pull.add_argument(
            "--format",
            type=str,
            default="json",
            help="Label format, choices: json (default) | mask"
        )

        ## dataset push
        dataset_push = dataset_action.add_parser("push", help="Push images to dataset")
        dataset_push.add_argument(
            "dataset",
            type=str,
            help="Name of dataset"
        )
        dataset_push.add_argument(
            "path",
            type=str,
            help="Path to image folder"
        )
        dataset_push.add_argument(
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