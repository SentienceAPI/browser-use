"""
CustomMessageManager: Simplified message manager for SentienceAgent.

Manages conversation history, agent history items, and message construction
without the complexity of the full MessageManager.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from browser_use.agent.message_manager.views import HistoryItem, MessageManagerState
from browser_use.llm.messages import BaseMessage, SystemMessage

if TYPE_CHECKING:
    from browser_use.agent.views import AgentOutput, AgentStepInfo, ActionResult

logger = logging.getLogger(__name__)


class CustomMessageManager:
    """
    Simplified message manager for SentienceAgent.

    Manages conversation history and agent history items without the
    full complexity of the standard MessageManager.
    """

    def __init__(
        self,
        task: str,
        system_message: SystemMessage,
        max_history_items: int | None = None,
    ):
        """
        Initialize CustomMessageManager.

        Args:
            task: The task for the agent
            system_message: System message for the LLM
            max_history_items: Maximum number of history items to keep (None = all)
        """
        self.task = task
        self.system_message = system_message
        self.max_history_items = max_history_items

        # Initialize state
        self.state = MessageManagerState()
        # Initialize with task (will be shown in agent_state, but include here for clarity)
        self.state.agent_history_items = [
            HistoryItem(step_number=0, system_message=f"<sys>Agent initialized. Task: {task}</sys>")
        ]

        # Store last messages for debugging
        self.last_input_messages: list[BaseMessage] = []

        logger.info(
            f"Initialized CustomMessageManager: task='{task}', "
            f"max_history_items={max_history_items}"
        )

    @property
    def agent_history_description(self) -> str:
        """
        Build agent history description from list of items.

        Respects max_history_items limit if set.

        Returns:
            Formatted history description string
        """
        if self.max_history_items is None:
            # Include all items
            return "\n".join(item.to_string() for item in self.state.agent_history_items)

        total_items = len(self.state.agent_history_items)

        # If we have fewer items than the limit, just return all items
        if total_items <= self.max_history_items:
            return "\n".join(item.to_string() for item in self.state.agent_history_items)

        # We have more items than the limit, so we need to omit some
        omitted_count = total_items - self.max_history_items

        # Show first item + omitted message + most recent (max_history_items - 1) items
        recent_items_count = self.max_history_items - 1  # -1 for first item

        items_to_include = [
            self.state.agent_history_items[0].to_string(),  # Keep first item (initialization)
            f"<sys>[... {omitted_count} previous steps omitted...]</sys>",
        ]
        # Add most recent items
        items_to_include.extend(
            [
                item.to_string()
                for item in self.state.agent_history_items[-recent_items_count:]
            ]
        )

        return "\n".join(items_to_include)

    def update_history(
        self,
        model_output: AgentOutput | None = None,
        result: list[ActionResult] | None = None,
        step_info: AgentStepInfo | None = None,
    ) -> None:
        """
        Update agent history with the latest step results.

        Args:
            model_output: Model output from LLM (if available)
            result: List of action results
            step_info: Step information
        """
        if result is None:
            result = []
        step_number = step_info.step_number if step_info else None

        # Clear read_state from previous step
        self.state.read_state_description = ""
        self.state.read_state_images = []

        # Process action results
        action_results = ""
        read_state_idx = 0

        for action_result in result:
            # Handle extracted content (one-time inclusion)
            if (
                action_result.include_extracted_content_only_once
                and action_result.extracted_content
            ):
                self.state.read_state_description += (
                    f"<read_state_{read_state_idx}>\n"
                    f"{action_result.extracted_content}\n"
                    f"</read_state_{read_state_idx}>\n"
                )
                read_state_idx += 1
                logger.info(
                    f"Added extracted_content to read_state_description: "
                    f"{action_result.extracted_content[:100]}..."
                )

            # Store images for one-time inclusion in the next message
            if action_result.images:
                self.state.read_state_images.extend(action_result.images)
                logger.info(f"Added {len(action_result.images)} image(s) to read_state_images")

            # Add to action results
            if action_result.long_term_memory:
                action_results += f"{action_result.long_term_memory}\n"
            elif (
                action_result.extracted_content
                and not action_result.include_extracted_content_only_once
            ):
                action_results += f"{action_result.extracted_content}\n"

            # Add errors
            if action_result.error:
                if len(action_result.error) > 200:
                    error_text = (
                        action_result.error[:100] + "......" + action_result.error[-100:]
                    )
                else:
                    error_text = action_result.error
                action_results += f"{error_text}\n"

        # Truncate read_state_description if too long
        MAX_CONTENT_SIZE = 60000
        if len(self.state.read_state_description) > MAX_CONTENT_SIZE:
            self.state.read_state_description = (
                self.state.read_state_description[:MAX_CONTENT_SIZE]
                + "\n... [Content truncated at 60k characters]"
            )
            logger.info("Truncated read_state_description to 60k characters")

        self.state.read_state_description = self.state.read_state_description.strip("\n")

        # Format action results
        if action_results:
            action_results = f"Result\n{action_results}"
        action_results = action_results.strip("\n") if action_results else None

        # Truncate action_results if too long
        if action_results and len(action_results) > MAX_CONTENT_SIZE:
            action_results = (
                action_results[:MAX_CONTENT_SIZE]
                + "\n... [Content truncated at 60k characters]"
            )
            logger.info("Truncated action_results to 60k characters")

        # Build the history item
        if model_output is None:
            # Add history item for initial actions (step 0) or errors (step > 0)
            if step_number is not None:
                if step_number == 0 and action_results:
                    # Step 0 with initial action results
                    history_item = HistoryItem(
                        step_number=step_number, action_results=action_results
                    )
                    self.state.agent_history_items.append(history_item)
                elif step_number > 0:
                    # Error case for steps > 0
                    history_item = HistoryItem(
                        step_number=step_number,
                        error="Agent failed to output in the right format.",
                    )
                    self.state.agent_history_items.append(history_item)
        else:
            # Normal step with model output
            history_item = HistoryItem(
                step_number=step_number,
                evaluation_previous_goal=model_output.current_state.evaluation_previous_goal
                if hasattr(model_output, "current_state")
                and hasattr(model_output.current_state, "evaluation_previous_goal")
                else None,
                memory=model_output.current_state.memory
                if hasattr(model_output, "current_state")
                and hasattr(model_output.current_state, "memory")
                else None,
                next_goal=model_output.current_state.next_goal
                if hasattr(model_output, "current_state")
                and hasattr(model_output.current_state, "next_goal")
                else None,
                action_results=action_results,
            )
            self.state.agent_history_items.append(history_item)

        logger.info(
            f"Updated history: step={step_number}, "
            f"history_items={len(self.state.agent_history_items)}"
        )

    def get_messages(
        self, user_message: BaseMessage | None = None
    ) -> list[BaseMessage]:
        """
        Get all messages for LLM call.

        Args:
            user_message: User message to include (if provided)

        Returns:
            List of messages in correct order: system -> user
        """
        messages = [self.system_message]
        if user_message:
            messages.append(user_message)
        return messages

    def add_new_task(self, new_task: str) -> None:
        """
        Add a new follow-up task to the conversation.

        Args:
            new_task: The new task to add
        """
        new_task_formatted = f"<follow_up_user_request> {new_task.strip()} </follow_up_user_request>"
        if "<initial_user_request>" not in self.task:
            self.task = f"<initial_user_request>{self.task}</initial_user_request>"
        self.task += "\n" + new_task_formatted

        task_update_item = HistoryItem(system_message=new_task_formatted)
        self.state.agent_history_items.append(task_update_item)

        logger.info(f"Added new task to conversation: {new_task[:50]}...")
