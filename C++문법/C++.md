# 정보은닉

구조체는 접근지정자를 정하지 않으면 public

클래스는 접근지정자를 정하지 않으면 private

private: 같은 클래스 내에서만 접근 가능

protected: 상속관계에 있었을 때 유도 클래스만 접근가능

public: 모든 클래스 접근 가능

# 멤버함수의 const 선언

ex) int GetX() const;  // 멤버변수의 값을 못바꿈

const 함수는 const가 아닌 함수를 호출하지 못함

const로 상수화된 객체를 대상으로는 const 멤버함수만 호출 가능

# 생성자

## 복사생성자

객체를 복사할 때 사용

```C++
Sosimple(const Sosimple & copy) //const가 꼭 들어가야됨
					: num(copy.num1), num2(copy.num2)
	{
                        
     }//default 복사생성자
```

복사 생성자가 호출되는 시점

1. 기존의 생성된 객체를 이용해서 새로운 객체를 초기화하는 경우
2. Call-by-value 방식의 함수호출 과정에서 객체를 인자로 전달하는 경우
3. 객체를 반환하되, 참조형으로 반환하지 않은 경우

```
Person man1("lee",32);
Person man2=man1 //복사 생성자 호출
```

복사생성자를 정의하지않으면 자동으로 디폴트 복사 생성자가 생김

explicit키워드를 사용하여 복사 생성자의 묵시적 호출을 허용하지 않도록 할 수 있다.

```C++
explicit Sosimple(const Sosimple & copy)
					: num(copy.num1), num2(copy.num2)
	{
                        //empty
     }
```

## 깊은 복사

디폴트 복사 생성자는 멤버 대 멤버의 단순 복사를 진행하기 때문에 복사의 결과로 하나의 문자열을 두개의 객체가 동시에 참조하는 꼴이 발생할 수 있다. 이로 인해 객체의 소멸과정에서 문제가 발생. 따라서 참조하는 문자열도 복사하여 두 개의 객체가 각각 문자열을 참조할 수 있게 하는 방법. 

# 소멸자

오버로딩 불가능, 인자값 없음 default소멸자는 아무일도 안함 

# 이니셜라이저 

:num1(n1)   ==> int num1 = n1 //선언과 동시에 초기화

num2 = n2 ==> int num2; num2=n2 // 두단계

# this포인터

this는 객체를 참조하는 포인터 자기 객체의 멤버변수를 가르킬 때 사용함 this->num

```C++
ex)
SelfRef& Adder(int n)
{
    num+=n;
    return *this;
}
```

객체 자신의 포인터가 아닌 객체 자신을 반환하겠다는 의미가 된다. 반환형이 참조형인 SelfRef&으로 선언되었으므로 객체 자신을 참조할 수 있는 참조값이 반환된다.

# friend와 static

friend 선언은 private 멤버의 접근을 허용하는 선언(역은 성립 x)

```c++
class Boy
{
  private:
    int height;
    friend class Girl;  //Girl클래스를 friend로 선언 hegith사용 가능
  public:
  	Boy(int len): height(len)
  	{ }
}
```

static 멤버

- 전역변수에 선언된 static의 의미
  - 선언된 파일 내에서만 참조를 허용하겠다는 의미
- 함수 내에 선언된 static의 의미
  - 한번만 초기화되고, 지역변수와 달리 함수를 빠져나가도 소멸되지 않는다.

static멤버변수

- 객체를 생성하건 생성하지 않건 메모리 공간에 딱 하나만 할당이 되어서 공유되는 변수
- public으로 선언되면 어디서든 접근가능

static멤버함수

- 선언된 클래스의 모든 객체가 공유한다.
- public으로 선언되면, 클래스의 이름을 이용해서 호출이 가능.
- 객체의 멤버로 존재하는 것이 아니다.

mutable키워드

- const함수 내에서의 값의 변경을 예외적으로 허용한다.

# 연산자 오버로딩

연산자 오버로딩: 객체끼리 연산을 할 수 있게하는 것

```C++
ex)

class Point{

private int x,y;

}

Point operator+(const Point &ref)

{

Point pos(x+ref.x, y+ref.y);

return pos;

}
```

pos1 + pos2;   ===> pos1.operator+(pos2);

연산자를 오버로딩하는 방법

1. 멤버함수에 의한 연산자 오버로딩
2. 전역함수에 의한 연산자 오버로딩

