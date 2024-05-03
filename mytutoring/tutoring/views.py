from django.shortcuts import render, get_object_or_404
from .models import Course, Lesson

def course_list(request):
    courses = Course.objects.all()
    return render(request, 'main.html', {'courses': courses})

def course_detail(request):
    # course = get_object_or_404(Course)
    return render(request, 'course-details.html')

def course_detail2(request):
    # course = get_object_or_404(Course)
    return render(request, 'course-details-2.html')

def course_detail3(request):
    # course = get_object_or_404(Course)
    return render(request, 'course-details-3.html')

def lesson_detail(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson.html')

def lesson_quiz(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-quiz.html')

def lesson_quiz_result(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-quiz-result.html')

def lesson_assign(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-assignments.html')

def lesson_assign_submit(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-assignments-submit.html')


def lesson_intro(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-intro.html')
