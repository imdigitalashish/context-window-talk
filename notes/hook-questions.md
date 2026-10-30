# Hook questions

Ask for a show of hands, then reveal the cause.

**Open:** "Before you hit Enter, can you predict whether your prompt will work?"
**Close:** "Now you can."

## Dumb zone law

- **Dumb zone:** "For the first 10 minutes your agent codes like a senior engineer. Two hours into the same chat, it forgets your folder structure. Who's seen that?"
- **Dumb zone:** "It's the same model, so why does it feel smarter in the chat app than inside your coding agent?"
- **Self-conditioning:** "You point out a bug, it says 'You're absolutely right!', and then writes the same bug again. Who's been stuck in that loop?"
- **Self-conditioning:** "Ever had a fresh chat fix in one shot what the old chat couldn't fix in 20 messages?"

## Clustering law

- **Scattered asks:** "You asked for frontend, backend, login and payments in one prompt. You got a bit of everything, and none of it worked. Hands up?"
- **Vocabulary:** "You said 'make it look better', and it changed random colours. A designer typed one sentence and got exactly what they wanted. Why?"
- **Negation blindness:** "You wrote 'DO NOT touch the tests' in capitals. What's the first thing it touched?"
- **Too many rules:** "Your rules file is 400 lines long. Honestly, how many of those rules does it follow?"
- **Literal matching:** "You described 'the function that handles payments' and it edited the wrong file. You pasted the exact name and it got it first try. Why?"

## U-shaped attention law

- **Lost in the middle:** "You pasted a 50-page PDF and asked a question. Did it answer from page 1, the last page, or page 25?" Reveal: the last page. In our 7B test at 32K, a fact at the end was found 80% of the time; at the start and in the middle, 0%.
- **Lost in the middle:** "In a long chat, what does it remember best?" The answer: what you just said.
- **Attention sinks (quiz):** "In this prompt, which token does the model look at most?" Reveal: a single newline in the chat template, not your words. In our 7B test it took 39% of the question's attention.
