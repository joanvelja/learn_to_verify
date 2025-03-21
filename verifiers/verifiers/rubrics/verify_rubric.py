from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

class VerifierRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.parser = XMLParser(fields=["reasoning", "answer"])
        self.reward_funcs = [
            self.exact_answer_reward_func,
            self.parser.get_xml_reward_func(),
            self.parser.get_format_reward_func(),
            self.starts_with_answer_reward_func,
        ]

    def correct_incorrect_answer_reward_func(self, completions, **kwargs) -> list[float]:
        '''Checks if the answer is either 'correct' or 'incorrect'
        Args:
            completions: list of completions
        Returns:
            list of rewards
        '''
        responses = [self.get_last_answer(c) for c in completions]
        return [1.0 if str(r) in ['correct', 'incorrect'] else 0.0 for r in responses]

    def starts_with_answer_reward_func(self, completions, **kwargs) -> list[float]:
        '''Checks if the answer starts with 'The answer is correct' or 'The answer is incorrect'.
        Ideally the model should not start like that, else it conditions the corresponding reasoning.
        Args:
            completions: list of completions
        Returns:
            list of rewards
        '''
        responses = [self.get_last_answer(c) for c in completions]
        # Check if the answer starts with 'The answer is correct' or 'The answer is incorrect', or "correct" or "incorrect"
        # return [1.0 if str(r).lower().startswith(('the answer is correct', 'the answer is incorrect', 'correct', 'incorrect')) else 0.0 for r in responses]
        # Possibly it's easier to enforce answer to start with <verification> tags
        return [0.1 if str(r).lower().startswith(('<verification>')) and not str(r).lower().startswith(('<verification>\ncorrect', '<verification>\nthe answer is correct', '<verification>\nincorrect', '<verification>\nthe answer is incorrect')) else 0.0 for r in responses]