# Pandas 객체

pandas란?

- pandas객체는 행과 열이 단순 정수형 인덱스가 아닌 레이블로 식별되는 Numpy의 구조화된 배열을 보강한 버전이라 볼 수 있음
- 기본 자료구조에 추가로 여러 가지 유용한 도구와 메서드 기능을 제공
- 세 가지 기본 자료 : Series, DataFrame, Index

## Pandas Series 객체

pandas series는 인덱싱된 데이터의 1차원 배열이다. Series는 일련의 값과 인덱스를 모두 감싸며 values와 index 속성으로 접근 가능 pandas series가 1차원 numpy배열보다 훨씬 더 일반적이고 유연함.

```python
 data = pd.Series([0.1, 0.2, 0.3])
 data
 
 결과 : 0 0.1
 	   1 0.2
 	   2 0.3
 	   3 0.4
 	   dtype: float64
```

### Series 객체 구성하기

```python

```



## Pandas DataFrame 객체

pandas dataFrame은 numpy 배열의 일반화 된 버전이나 파이썬 딕셔너리의 특수한 버전

DataFrame은 유연한 행 인덱스와 유연한 열 이름을 가진 2차원 배열, 정렬(index를 공유한)된  Series 객체의 연속 

```python
population = {'a' : 1000, 'b' : 2000, 'c' ; 3000 }
population = pd.Series(area_dict)
area = {'a' : 100, 'b' : 200, 'c' ; 300 }
area = pd.Series(area_dict)
data = pd.DataFrame({'population': population, 'area' = area})
data
결과값       area     population
	a       100			1000
	b	    200			2000
	c	    300			3000
	dtype: int64
```

## Pandas Index 객체

Index 객체는 불변의 배열이나 정렬된 집합(Index 객체가 중복되는 값을 포함할 수 있으므로 기술적으로 중복집합)이다.

### Index 불변의 배열

index 객체는 여러 면에서 배열로 동작 파이썬 인덱싱 표기법을 사용해 값이나 슬라이스를 가져올 수 있다.

numpy배열과의 한가지 차이점은 Index객체는 일반적인 방법으로 변경될 수 없다.

```python
ind = pd.Index([1,2,3,4,5])
ind

결과: Int64Index([1,2,3,4,5], dtype='int64')

ind[1]
결과 2
ind[::2]
결과 Int64Index([1,3,5], dtype='int64')

```



# 데이터 인덱싱과 선택

Series: 딕셔너리 

- Series 객체는 딕셔너리와 마찬가지로 키의 집합을 값의 집합에 매핑한다.
- 키/인덱스와 값을 조사하기 위해 딕셔너리와 유사한 파이썬 표현식과 메서드를 사용할 수도 있다.
- 딕셔너리와 유사한 구문을 사용해 수정할 수도 있다. 새로운 키에 할당해 딕셔너리를 확장할 수 있는 것과 마찬가지로 새로운 인덱스 값에 할당함으로써 Series를 확장할 수 있다.

Series: 1차원 배열

- 슬라이스, 마스킹, 팬시 인덱싱등 numpy 배열과 같은 메커니즘으로 배열 형태의 아이템을 선택할 수 있음.

인덱서: loc, iloc, ix

- 이 슬라이싱과 인덱싱의 표기법은 표기법에서 혼동됨 이 문제를 해결하기위해 인덱서(Series의 데이터에 대한 특정 슬라이싱 인터페이스를 드러내는 속성)를 제공
- loc 속성은 언제나 명시적인 인덱스를 참조하는 인덱싱과 슬라이싱을 가능하게 만듬
- iloc 속성은 인덱싱과 슬라이싱에서 파이썬 스타일의 인덱스를 참조하게 해줌

DataFrame: 딕셔너리

DataFrame: 2차원 배열

- DataFrame 객체 인덱싱에서 열을 딕셔너리 스타일로 인덱싱하면 그 객체를 단순히 numpy배열로 다룰 수 없게 된다.  

  

# Pandas에서 연산

pandas는 numpy로 부터 상속받고 유니버설 함수가 그 핵심임.

pandas를 이용하면 데이터의 맥락을 유지하고 다른 소스에서 가져온 데이터를 결합하는 작업을 실패할 일이 없게 만듬.