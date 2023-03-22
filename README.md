# dl-coders-torch-book

```mermaid
flowchart LR

A(init) -->B(predict)
B --> C(loss)
C --> D(gradient)
D --> F(Step)
F --> G(stop)
B --> F
```