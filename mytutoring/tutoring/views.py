from django.shortcuts import render, get_object_or_404
from .models import Course, Lesson
from django.contrib.auth.decorators import login_required

def login(request):
    return render(request, 'login.html')

@login_required(login_url='/login/')
def dashboard(request):
    return render(request, 'student-dashboard.html')

def course_list(request):
    courses = Course.objects.all()
    return render(request, 'main.html', {'courses': courses})

@login_required(login_url='/login/')
def course_detail(request):
    # course = get_object_or_404(Course)
    return render(request, 'course-details.html')

@login_required(login_url='/login/')
def course_detail2(request):
    # course = get_object_or_404(Course)
    return render(request, 'course-details-2.html')

@login_required(login_url='/login/')
def course_detail3(request):
    # course = get_object_or_404(Course)
    return render(request, 'course-details-3.html')

@login_required(login_url='/login/')
def lesson_detail(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson.html')

@login_required(login_url='/login/')
def lesson_quiz(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-quiz.html')

@login_required(login_url='/login/')
def lesson_quiz_result(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-quiz-result.html')

@login_required(login_url='/login/')
def lesson_assign(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-assignments.html')

@login_required(login_url='/login/')
def lesson_assign_submit(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-assignments-submit.html')

@login_required(login_url='/login/')
def lesson_intro(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'lesson-intro.html')

@login_required(login_url='/login/')
def student_course(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'student-enrolled-courses.html')

@login_required(login_url='/login/')
def student_quiz(request):
    # lesson = get_object_or_404(Lesson, id=id)
    return render(request, 'student-my-quiz-attempts.html')
