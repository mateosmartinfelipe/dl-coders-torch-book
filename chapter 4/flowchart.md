# dl-coders-torch-book

## SGD

```mermaid
flowchart LR

A(init) -->B(predict)
B --> C(loss)
C --> D(gradient)
D --> F(Step)
F --> G(stop)
B --> F
```
