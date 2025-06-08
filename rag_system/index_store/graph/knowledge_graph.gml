graph [
  directed 1
  node [
    id 0
    label "Ollama"
    type "Company"
    properties [
    ]
  ]
  node [
    id 1
    label "Llama3"
    type "Product"
    properties [
    ]
  ]
  node [
    id 2
    label "nomic-embed-text"
    type "Product"
    properties [
    ]
  ]
  node [
    id 3
    label "tim_cook"
    type "Person"
    properties [
    ]
  ]
  node [
    id 4
    label "apple"
    type "Company"
    properties [
    ]
  ]
  edge [
    source 0
    target 1
    label "HAS_MODEL"
  ]
  edge [
    source 0
    target 2
    label "HAS_MODEL"
  ]
  edge [
    source 4
    target 3
    label "CEO"
  ]
]
