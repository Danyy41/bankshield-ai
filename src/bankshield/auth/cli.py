"""Provision user accounts -- the only way to create users or change roles.

There is intentionally no HTTP endpoint for this: account management needs
operator access to the deployment (shell / `docker exec` / an admin task),
which keeps it out of reach of anything that can only call the API -- an
MCP client, an A2A agent, or a stolen analyst token.

    python -m bankshield.auth.cli create-user alice --role reviewer   # prompts for a password
    echo "$PASSWORD" | python -m bankshield.auth.cli create-user svc-triage --role agent --password-stdin
    python -m bankshield.auth.cli disable-user alice
    python -m bankshield.auth.cli list-users
"""

from __future__ import annotations

import argparse
import getpass
import sys

from bankshield.auth.users import ROLES, UserStore
from bankshield.persistence import audit
from bankshield.persistence.db import get_database


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        sys.exit("passwords do not match")
    return first


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bankshield.auth.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-user")
    create.add_argument("username")
    create.add_argument("--role", required=True, choices=ROLES)
    create.add_argument("--password-stdin", action="store_true")
    for name in ("disable-user", "enable-user"):
        sub.add_parser(name).add_argument("username")
    reset = sub.add_parser("set-password")
    reset.add_argument("username")
    reset.add_argument("--password-stdin", action="store_true")
    sub.add_parser("list-users")
    args = parser.parse_args(argv)

    store = UserStore()
    try:
        if args.command == "create-user":
            user = store.create(args.username, _read_password(args.password_stdin), args.role)
            detail = {"role": user.role}
            print(f"created {user.username} ({user.role})")
        elif args.command in ("disable-user", "enable-user"):
            store.set_disabled(args.username, args.command == "disable-user")
            detail = {}
            print(f"{args.command.split('-')[0]}d {args.username}")
        elif args.command == "set-password":
            store.set_password(args.username, _read_password(args.password_stdin))
            detail = {}
            print(f"password updated for {args.username}")
        else:
            for user in store.list():
                print(f"{user.username}\t{user.role}\t{'disabled' if user.disabled else 'active'}")
            return
    except (ValueError, KeyError) as exc:
        sys.exit(f"error: {exc}")
    audit.record_event(get_database(), action=f"admin.{args.command}", outcome="ok", actor="cli",
                       target=args.username, detail=detail)


if __name__ == "__main__":
    main()
