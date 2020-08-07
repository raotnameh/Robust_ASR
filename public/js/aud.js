URL = window.URL || window.webkitURL;

var strm; 						//stream from getUserMedia()
var rec; 					        //Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording
// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext; //audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");


//add events to those 2 buttons
recordButton.addEventListener("click", start);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pause);

function start(){
    recordButton.disabled = true;
    stopButton.disabled = false;
    pauseButton.disabled = false;
    console.log("starting to record");

    navigator.mediaDevices.getUserMedia({audio:true,video:false}).then(function(stream){
       strm = stream;
       audioContext = new AudioContext();
       input = audioContext.createMediaStreamSource(stream);
       rec = new Recorder(input,{numChannels:1});
       rec.record();
       console.log("recording");
    }).catch(function (err){
        recordButton.disabled = false;
    	stopButton.disabled = true;
    	pauseButton.disabled = true;
    });
}
function pause(){
    console.log("stopping");
    if (rec.recording){
        rec.stop();
        pauseButton.innerHTML = "Resume";
    }
    else{
        rec.record();
        pauseButton.innerHTML = "Pause";
    }
}

function stopRecording() {
	console.log("stopButton clicked");

	//disable the stop button, enable the record too allow for new recordings
	stopButton.disabled = true;
	recordButton.disabled = false;
	pauseButton.disabled = true;

	//reset button just in case the recording is stopped while paused
	pauseButton.innerHTML="Pause";
	
	//tell the recorder to stop the recording
	rec.stop();

	//stop microphone access
	strm.getAudioTracks()[0].stop();

	//create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}


function createDownloadLink(blob) {
	
	var url = URL.createObjectURL(blob);
	var detailList = document.createElement("ul");
	var textList = document.createElement("ul");
	detailList.setAttribute("style","list-style-type: none")
	textList.setAttribute("style","list-style-type: none")
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var li2 = document.createElement('li');
	var li3 = document.createElement("li");
	var link = document.createElement('a');
	var predicted = document.createElement("input")
	var inpt = document.createElement("input")
	var country = document.createElement("select")
	var gender = document.createElement("select")
	var age = document.createElement("input")
	age.setAttribute("id","age")
	age.setAttribute("type","number")
	age.setAttribute("min","1")
	age.placeholder = "Age"
	gender.setAttribute("id","gender")
	country.setAttribute("id","country")
	inpt.setAttribute("id","spoken")
	inpt.placeholder = "True Text"
	predicted.setAttribute("id","predicted")
	predicted.placeholder = "Predicted Text"
	var genderArr = new Array("Male","Female")
	var country_arr = new Array("Afghanistan", "Albania", "Algeria", "American Samoa", "Angola", "Anguilla", "Antartica", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Ashmore and Cartier Island", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "British Virgin Islands", "Brunei", "Bulgaria", "Burkina Faso", "Burma", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China", "Christmas Island", "Clipperton Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Cook Islands", "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czeck Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Europa Island", "Falkland Islands (Islas Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "French Guiana", "French Polynesia", "French Southern and Antarctic Lands", "Gabon", "Gambia, The", "Gaza Strip", "Georgia", "Germany", "Ghana", "Gibraltar", "Glorioso Islands", "Greece", "Greenland", "Grenada", "Guadeloupe", "Guam", "Guatemala", "Guernsey", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Heard Island and McDonald Islands", "Holy See (Vatican City)", "Honduras", "Hong Kong", "Howland Island", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Ireland, Northern", "Israel", "Italy", "Jamaica", "Jan Mayen", "Japan", "Jarvis Island", "Jersey", "Johnston Atoll", "Jordan", "Juan de Nova Island", "Kazakhstan", "Kenya", "Kiribati", "Korea, North", "Korea, South", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Macedonia, Former Yugoslav Republic of", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Man, Isle of", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia, Federated States of", "Midway Islands", "Moldova", "Monaco", "Mongolia", "Montserrat", "Morocco", "Mozambique", "Namibia", "Nauru", "Nepal", "Netherlands", "Netherlands Antilles", "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcaim Islands", "Poland", "Portugal", "Puerto Rico", "Qatar", "Reunion", "Romainia", "Russia", "Rwanda", "Saint Helena", "Saint Kitts and Nevis", "Saint Lucia", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Scotland", "Senegal", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia and South Sandwich Islands", "Spain", "Spratly Islands", "Sri Lanka", "Sudan", "Suriname", "Svalbard", "Swaziland", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Tobago", "Toga", "Tokelau", "Tonga", "Trinidad", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "Uruguay", "USA", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Virgin Islands", "Wales", "Wallis and Futuna", "West Bank", "Western Sahara", "Yemen", "Yugoslavia", "Zambia", "Zimbabwe");
	var curCount;
	for (var i =0;i<country_arr.length;i++){
		curCount = document.createElement("option")
		curCount.setAttribute("id",country_arr[i])
		curCount.innerHTML = country_arr[i]
		country.appendChild(curCount)
		if (country_arr[i] == "India"){
			country.selectedIndex = i;
		}
	}
	var curGend;
	for (var i =0;i<genderArr.length;i++){
		curGend = document.createElement("option")
		curGend.setAttribute("id",genderArr[i])
		curGend.innerHTML = genderArr[i]
		gender.appendChild(curGend)
	}

	var filename = new Date().toISOString();

	//add controls to the <audio> element
	au.controls = true;
    au.src = url;
    console.log(url)

	//save to disk link
	link.href = url;
	link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
	link.innerHTML = "Download";

	
	//add the new audio element to li
	li.appendChild(au);
	
	li3.appendChild(inpt)
	li2.appendChild(country)
	li2.appendChild(gender)
	li2.appendChild(age)
	li3.appendChild(predicted);
	//add the save to disk link to li
	li.appendChild(link);
	

	//upload link
	var upload = document.createElement('a');
	upload.href="#";
	upload.innerHTML = "Upload";
	upload.addEventListener("click", function(event){
		  var xhr=new XMLHttpRequest();
		  xhr.onload=function(e) {
		      if(this.readyState === 4) {
		          console.log("Server returned: ",e.target.responseText);
		      }
		  };
		  var fd=new FormData();
		  console.log(inpt.value)
		//   fd.append("audioText",inpt.value)
		  sendData = {
			  "audioText": inpt.value,
			  "country":country.value,
			  "gender":gender.value,
			  "age":age.value
		  }
		  $.each(sendData, function(key, value){
			fd.append(key, value);
		  })
		  fd.append("file",blob, filename);

		  xhr.open("POST","/",true);
		  xhr.send(fd);
		  xhr.onload=()=>{
			var  son = xhr.responseText;
			// predicted.innerHTML = son;
			var rep = JSON.parse(xhr.responseText);
			predicted.value = rep.name
			
		  }
	})
	li.appendChild(document.createTextNode (" ")); //add a space in between
	li.appendChild(upload); 	//add the upload link to li



	recordingsList.appendChild(li);
	detailList.appendChild(li2)
	textList.appendChild(li3)
	recordingsList.appendChild(detailList)
	recordingsList.appendChild(textList)
}