## Key principals for writing code

- Keep functions short and focused on one responsibility.
- Prefer clear, readable code over clever shortcuts as long as runtime is not impacted.
- Add short comments only when behavior is not obvious.
- Organize related logic using classes.
- Group related variables using dataclasses; avoid many loose locals or excessive use of self._ attributes.
- Apply strong type hints throughout the code and use assert statements for validation.
- Separate business logic from boilerplate by using decorators and context managers—use this for things like monitoring setup or dependency injection.
- Replace busy-wait loops with event-driven synchronization where possible.
- Ensure each module has a single, clear responsibility.
- Configure a shared logger for each package. Keep logs from external sources (such as vLLM) in separate loggers.