## PEP8

- 세로로 정렬시켜주어야한다.

```python
foo = long_function_name(var_one, var_two,
                         var_three, var_four)
```

- 모든 줄을 79자로 제한한다.

```python
with open('/path/to/some/file/you/want/to/read') as file_1, \
     open('/path/to/some/file/being/written', 'w') as file_2:
    file_2.write(file_1.read())
```

- 연산자는 아래에 붙여쓴다.

```python
income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest)
```

- 공백

가장 높은 위치에 있는 함수와 클래스 사이에 2개의 공백 line을 사용한다.
쓸모없는 space는 줄인다.

- 인코딩

`utf-8`

- 한줄에 import 하나씩 하기

```python
import os
import sys
```

- 이름 명명법
```
ClassName
ExceptionName
module_name
package_name
method_name
function_name
function_parameter_name
global_var_name
local_var_name
instance_var_name

GLOBAL_CONSTANT_NAME
```

# REFERENCE
- [https://www.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds](https://www.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds)
