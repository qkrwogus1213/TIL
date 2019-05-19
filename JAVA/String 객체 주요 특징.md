# Object 클래스

## 객체를 문자열로 변환

```java
class Point {
private int x, y;
public Point(int x, int y) {
this.x = x;
this.y = y;
}
/*
public String toString() {
return "Point(" + x + "," + y + ")";  
}
클래스를 만들때 toString을 오버라딩해라.
안해주면 System.out.println(obj); 에서 @가 생김*/
}
public class ObjectPropertyEx {
public static void print(Object obj) {
System.out.println(obj.getClass().getName()); // 클래스 이름// obj.getClass()는 Point클래스를 가르킴 .getName()은 리턴 타입이 문자열 가르키고있는 클래스이름 출력
System.out.println(obj.hashCode()); // 해시 코드 값
System.out.println(obj.toString()); // 객체를 문자열로 만들어 출력
System.out.println(obj); // 객체 출력 .toString()호출
}
public static void main(String [] args) {
Point p = new Point(2,3);
print(p);
}
}

/*
출력:
Point
366712642
Point@15db9742
Point@15db9742
*/
```

## String to String()

- 객체를 문자열로반환

  ```java
  public String toString() {
  return getClass().getName() +"@" + Integer.toHexString(hashCode());
  }
  ```

## 객체 비교와 equals()

```java
Point a = new Point(2,3);
Point b = new Point(2,3);
Point c = a;
if(a == b) // false
System.out.println("a==b");
if(a == c) // true
System.out.println("a==c");
```

2.함수 삽입

```java
class Point {
int x, y;
public Point(int x, int y) {
this.x = x; this.y = y;
}
public boolean equals(Object p) { //멤버값들 비교 //오버라이딩 해서 사용해라.
Point p = (Point)obj;
if(x == p.x && y == p.y)
return true;
else return false;
}
}
```

```java
Point a = new Point(2,3);
Point b = new Point(2,3);
Point c = new Point(3,4);
if(a == b) // false
System.out.println("a==b");
if(a.equals(b)) // true
System.out.println("a is equal to b");
if(a.equals(c)) // false
System.out.println("a is equal to c");
```

## Wrapper Class

기본 타입의 값을 객체로 다룰 수 있게 함

1.기본 타입 값으로 생성

```java
Integer i = Integer.valueOf(10);
Character c = Character.valueOf(‘c’);
Double f = Double.valueOf(3.14);
Boolean b = Boolean.valueOf(true);
/*
Integer i = new Integer(10);
Character c = new Character(‘c’);
Double f = new Double(3.14);
Boolean b = new Boolean(true);
Java 9부터 생성자를 이용한 Wrapper 객체 생성 불가 warning뜸
*/
```

2,문자열로 생성

```java
Integer I = Integer.valueOf(“10”);
Double d = Double.valueOf(“3.14”);
Boolean b = Boolean.valueOf(“false”);
/*
Integer I = new Integer(“10”);
Double d = new Double(“3.14”);
Boolean b = new Boolean(“false”);
Java 9부터 생성자를 이용한 Wrapper 객체 생성 불가 warning뜸
*/
```

3.Wrapper객체로 부터 기본 타입 값 알아내기

```java
Integer i = Integer.valueOf(10);
int ii = i.intValue(); // ii = 10
```

## 박싱과 언박싱

1. 박싱
   - 기본 타입의 값을 Wrapper 객체로 변환
2. 언박싱
   - Wrapper객체에 들어 있는 기본타입의 값을 빼내는 것

```java
Integer ten = 10; // 자동 박싱. Integer ten = Integer.valueOf(10);로 자동 처리
int n = ten; // 자동 언박싱. int n = ten.intValue();로 자동 처리
```

## 스트링 리터럴과 new String()차이

리터럴로 생성: JVM이 리터럴 관리, 응용프로그램 내에서 공유됨

```java
String a= "h";
String b ="j";
String c = "h"; // a가 가르키는 "h"를 가르킴
```

new String(): 힙 메모리에 String 객체 각각 생성

String 특징

- 수정 불가능
- 비교시 equals() 사용
- concat()은 새로운 문자열 생성
- String trim(): 공백 제거

## StringBuffer 클래스 

가변 크기의 문자열 저장 클래스

선언 방식:  StringBuffer sb = new StringBuffer("java");

## StringTokenizer클래스

하나의 문자열을 여러 문자열 분리

```java
String query = "name=kitae&addr=seoul&age=21";
StringTokenizer st = new StringTokenizer(query, "&");
```

