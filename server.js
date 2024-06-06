function sendRequest() {
    var responseContainer = document.getElementById('responses');
    var serverResponseElement = document.getElementById('server_res');
    var recc=document.getElementById('recommendations');
    var songName = document.getElementById('songName').value;
    responseContainer.style.display='none';
    recc.style.display='none';
    // console.log('yoyo');
    fetch('http://127.0.0.1:5000/process_name', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name: songName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.result) {
            serverResponseElement.innerText = 'Genre : ' + data.result;
        } else if (data.error) {
            serverResponseElement.innerText = 'Error: ' + data.error;
        } else {
            serverResponseElement.innerText = 'Unexpected server response: ' + JSON.stringify(data);
        }
        responseContainer.style.display='block';
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
function upload(event) {
    event.preventDefault(); // Prevent the default form submission behavior
    var upload_win = document.getElementById('to_upload');
    upload_win.style.display = 'block';
    upload_win.style.transition="transform 0.3s ease-in-out";
}
function defaulty(){
    var upload_win = document.getElementById('to_upload');
    upload_win.style.display = 'none';
}

function uploadAudio() {
    var audioFileInput = document.getElementById('audioFile');
    var audioFile = audioFileInput.files[0];

    if (audioFile) {
        var formData = new FormData();
        formData.append('audioFile', audioFile);

        fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Server Response:', data);
            show(data);
            defaulty();
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        console.error('No audio file selected.');
    }
}
function show(data)
{
    var responseContainer = document.getElementById('responses');
    var serverResponseElement = document.getElementById('server_res');
    responseContainer.style.display='none';
    if (data.Genre) {
        serverResponseElement.innerText = 'Genre : ' + data.Genre;
    } else if (data.error) {
        serverResponseElement.innerText = 'Error: ' + data.error;
    } else {
        serverResponseElement.innerText = 'Unexpected server response: ' + JSON.stringify(data);
    }
    responseContainer.style.display='block';
}
function recommend()
{
    var func=document.getElementById('recommendations')
    var songName = document.getElementById('songName').value;
    // console.log('yoyo');
    if(songName)
    {
        fetch('http://127.0.0.1:5000/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: songName })
        })
        .then(response => response.json())
        .then(data => {
            for(var i=1;i<=6;i++)
            {
                var img=document.getElementById('img'+i.toString());
                img.src=data[i-1][2];
                var name=document.getElementById('name'+i.toString());
                name.innerHTML=data[i-1][0];
                var artist=document.getElementById('artist'+i.toString());
                artist.innerHTML=data[i-1][1];
                var link=document.getElementById('link'+i.toString());
                link.href=data[i-1][3];
            }
            func.style.display='block'
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    else
    {
        fetch('http://127.0.0.1:5000/recommend_audio', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            for(var i=1;i<=6;i++)
            {
                var img=document.getElementById('img'+i.toString());
                img.src=data[i-1][2];
                var name=document.getElementById('name'+i.toString());
                name.innerHTML=data[i-1][0];
                var artist=document.getElementById('artist'+i.toString());
                artist.innerHTML=data[i-1][1];
                var link=document.getElementById('link'+i.toString());
                link.href=data[i-1][3];
            }
            func.style.display='block'
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}