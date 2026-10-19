"""Task lifecycle shared by both agent executors: create the A2A task, run
the agent's work, publish its result as a data artifact, and map failures
onto A2A task states without leaking internals."""

from __future__ import annotations

import json
from typing import Awaitable, Callable

import a2a.types as a2a_types
from a2a.helpers import new_data_part, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from bankshield.investigation import data_access
from bankshield.security.guardrails import sanitize_error_response
from bankshield.security.validators import ToolInputValidationError


async def run_task(
    context: RequestContext,
    event_queue: EventQueue,
    work: Callable[[RequestContext], Awaitable[dict]],
    *,
    artifact_name: str,
) -> None:
    task = context.current_task
    if task is None:
        # Built directly (not via new_task_from_user_message, which raises on
        # a malformed message) so even a bad request gets a task we can reject.
        task = a2a_types.Task(
            id=context.task_id,
            context_id=context.context_id,
            status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_SUBMITTED),
            history=[context.message] if context.message else [],
        )
        await event_queue.enqueue_event(task)
    updater = TaskUpdater(event_queue, task.id, task.context_id)

    def reply(text: str):
        return updater.new_agent_message([new_text_part(text)])

    await updater.start_work()
    try:
        result = await work(context)
    except ToolInputValidationError as exc:
        # Malformed or unexpected input: refuse the task outright.
        await updater.reject(reply(str(exc)))
        return
    except data_access.NotFoundError as exc:
        await updater.failed(reply(str(exc)))
        return
    except Exception as exc:  # fail closed: log detail, return a generic message
        await updater.failed(reply(sanitize_error_response(exc)["detail"]))
        return

    # Round-trip through JSON so the artifact holds plain JSON types only.
    payload = json.loads(json.dumps(result, default=str))
    await updater.add_artifact([new_data_part(payload)], name=artifact_name)
    await updater.complete()
