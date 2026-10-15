#задание1
class Person:
    def __init__(self, full_name: str):
        self.full_name = full_name

    def get_name(self) -> str:
        return self.full_name

    def set_name(self, name: str) -> None:
        self.full_name = name

    def display(self) -> None:
        print(f"person: {self.full_name}")


class Student(Person):
    def __init__(self, full_name: str, course: int, group: str):
        super().__init__(full_name)
        self.course = course
        self.group = group

    def get_course(self) -> int:
        return self.course

    def set_course(self, course: int) -> None:
        self.course = course

    def get_group(self) -> str:
        return self.group

    def set_group(self, group: str) -> None:
        self.group = group

    def display(self) -> None:
        print(f"student: {self.full_name}, course: {self.course}, group: {self.group}")


class Professor(Person):
    def __init__(self, full_name: str, department: str):
        super().__init__(full_name)
        self.department = department

    def get_department(self) -> str:
        return self.department

    def set_department(self, department: str) -> None:
        self.department = department

    def display(self) -> None:
        print(f"professor: {self.full_name}, department: {self.department}")



def main():
    s1 = Student("bob", 2, "ghjjjj")
    p1 = Professor("bob2", "lkj")
    all_people = [s1, p1]
    for person in all_people:
        person.display()

   


if __name__ == "__main__":
    main()