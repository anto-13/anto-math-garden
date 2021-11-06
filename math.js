var answer;
var score = 0;
var backgroundImages = [];

function nextQuestion() {
    const n1 = Math.floor(Math.random() * 5);
    const n2 = Math.floor(Math.random() * 6);

    document.getElementById('n1').innerHTML = n1;
    document.getElementById('n2').innerHTML = n2;
    answer = n1 + n2;
}

function checkAnswer() {

    const prediction = predictImage();
    console.log(`answer: ${answer}, prediction: ${prediction}`);

    if (prediction == answer) {
        score++;
        if (score < 7) {
            console.log(`Correct. Score ${score}`);
            backgroundImages.push(`url('images/background${score}.svg')`);
            document.body.style.backgroundImage = backgroundImages;
        } else {
            alert('Congratulations! You ended up the quiz, Want to start again?');
            score = 0;
            backgroundImages = [];
            document.body.style.backgroundImage = backgroundImages;
        }
    } else {
        if (score != 0) { score--; }

        alert('Wrong answer, the garden is dying!');
        setTimeout(function () {
            backgroundImages.pop(`url('images/background${score}.svg')`);
            document.body.style.backgroundImage = backgroundImages;
        }, 2000)
    }
}