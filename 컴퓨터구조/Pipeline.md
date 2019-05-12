# 파이프 라인

파이프라이닝: 여러 명령어가 중첩되어 실행되는 구현기술

파이프라이닝에 의한 속도 향상은 파이프라인의 단계 수와 같다.

MIPS명령어는 다섯 단계가 걸린다

1. 메모리에서 명령어를 가져온다
2. 명령어를 해독하는 동시에 레지스터를 읽는다. MIPS명령어는 형식이 규칙적이므로 읽기와 해독이 동시에 일어날 수 있다.
3. 연산을 수행하거나 주소를 계산한다.
4. 데이터 메모리에 있는 피연산자에 접근한다.
5. 결과 값을 레지스터에 쓴다.

명령어 사이의 시간(파이프라인) = 명령어 사이의 시간(파이프라이닝되지 않음) / 파이프 단계 수

# Data Hazard

어떤 단계가 다른 단계가 끝나기를 기다려야 하기 때문에 파이프 라인이 지연되어야 하는 경우(계획된 명령어가 적절한 클럭 사이클에 실행될 수 없는 사건)

# Data Hazard 발생 조건

Ex/Mem.rd = ID/Ex.rs    -> 1a

Ex/Mem.rd = ID/Ex.rt 	  -> 1b

MEM/WB.rd = ID/EX.rs  -> 2a

MEM/WB.rd = ID/EX.rt   -> 2b



# Hazard 해결 방법

## 1. Stalling

## 2. Insert nop Instruction

nop  //  no operation 

bubble 대신 nop을 집어넣음 // 명령어에서 bubble 대신 nop을 사용

## 3. Code Scheduling

순서가 바꿔도 상관없는 명령어를 채움

## 4. Forwarding

add -> $s0 -> sub 이지만 Forwarding을 쓰면 레지스터를 거치지않고 add->sub가된다.

# Forwarding으로 해결불가

load명령어이고 그뒤에 사용하는 명령어가 뒤따라 올 때

ID/EX.rt = IF/ID.rs 또는 ID/EX.rt = IF/ID.rt

해결 방법: 한클락을 쉬어야됨
