// set our variables
var alarmTime, setting_alarm, alarm,
    activeAlarm = false,
    sound = new Audio("https://freesound.org/data/previews/316/316847_4939433-lq.mp3");


var h_alarm = "1"
var m_alarm = "0"
var s_alarm = "0"
var session_alarm = "AM"

const hourHand = document.querySelector('[data-hour-hand]')
const minuteHand = document.querySelector('[data-minute-hand]')
const secondHand = document.querySelector('[data-second-hand]')

function setClock() {
  const date = new Date();
  const secondsRatio = date.getSeconds() / 60;
  const minutesRatio = (secondsRatio + date.getMinutes()) / 60;
  const hoursRatio = (minutesRatio + date.getHours()) / 12;

  setRotation(secondHand, secondsRatio);
  setRotation(minuteHand, minutesRatio)
  setRotation(hourHand, hoursRatio)

  setTimeout(setClock, 1000)
}

function setRotation(element, rotationRatio) {
  element.style.setProperty('--deg', rotationRatio * 360)
}

setClock();

function showTime(){
    var date = new Date();
    var h = date.getHours(); // 0 - 23
    var m = date.getMinutes(); // 0 - 59
    var s = date.getSeconds(); // 0 - 59
    var session = "AM";

    if(h == 0){
        h = 12;
    }

    if(h > 12){
        h = h - 12;
        session = "PM";
    }

    h = (h < 10) ? "0" + h : h;
    m = (m < 10) ? "0" + m : m;
    s = (s < 10) ? "0" + s : s;

    var time = h + ":" + m + ":" + s + " " + session;

    if (setting_alarm == true) {

      if (h_alarm.length < 2) {
        h_alarm = (h_alarm < 10) ? "0" + h_alarm : h_alarm;
      }
      if (m_alarm.length < 2) {
        m_alarm = (m_alarm < 10) ? "0" + m_alarm : m_alarm;
      }
      if (s_alarm.length < 2) {
        s_alarm = (s_alarm < 10) ? "0" + s_alarm : s_alarm;
      }

      alarmTime = h_alarm + ":" + m_alarm + ":" + s_alarm + " " + session_alarm;
      document.getElementById("display").innerText = alarmTime;
      document.getElementById("display").textContent = alarmTime;
    } else {
      document.getElementById("display").innerText = time;
      document.getElementById("display").textContent = time;

    }
    setTimeout(showTime, 1000);

    if (activeAlarm == true) {
      if (time === alarmTime) {
        sound.play();
        alarm = true;
      }
    }

}

showTime();

var setting_alarm = false;

function setAlarmTime(element) {

    if (setting_alarm == false) {


      setting_alarm = true;
      if (h_alarm.length < 2) {
        h_alarm = (h_alarm < 10) ? "0" + h_alarm : h_alarm;
      }
      if (m_alarm.length < 2) {
        m_alarm = (m_alarm < 10) ? "0" + m_alarm : m_alarm;
      }
      if (s_alarm.length < 2) {
        s_alarm = (s_alarm < 10) ? "0" + s_alarm : s_alarm;
      }

      var alarmTime = h_alarm + ":" + m_alarm + ":" + s_alarm + " " + session_alarm;

    } else {
      setting_alarm = false;
    }


}

function Confirm() {
    if (setting_alarm == true) {
      activeAlarm = true;
      setting_alarm = false;
      var row = document.getElementsByClassName('alarm alarm_marker')
      for ( var i=0; i<row.length; i++ ) {
        row[i].style.backgroundColor = "red";
      }

    }

}

function increaseHours() {
    if (setting_alarm == true) {
      h_alarm = parseInt(h_alarm) + 1;
      if (h_alarm > 12) {
        h_alarm = 1;
        if (session_alarm == "AM") {
          session_alarm = "PM";
        } else {
          session_alarm = "AM";
        }
      }
      h_alarm = h_alarm.toString()
      showTime();
      setClock();
    }

}

function decreaseHours() {
    if (setting_alarm == true) {
      h_alarm = parseInt(h_alarm) - 1;
      if (h_alarm < 1) {
        h_alarm = 12;
        if (session_alarm == "AM") {
          session_alarm = "PM";
        } else {
          session_alarm = "AM";
        }
      }
      h_alarm = h_alarm.toString()
      showTime();
      setClock();
    }

}

function increaseMinutes() {
    if (setting_alarm == true) {
      m_alarm = parseInt(m_alarm) + 1;
      if (m_alarm > 59) {
        m_alarm = 0;
        h_alarm = h_alarm + 1;
        if (h_alarm > 12) {
          h_alarm = 1;
          if (session_alarm == "AM") {
            session_alarm = "PM";
          } else {
            session_alarm = "AM";
          }
        }
      }
      m_alarm = m_alarm.toString();
      h_alarm = h_alarm.toString();
      showTime();
      setClock();
    }

}

function decreaseMinutes() {
    if (setting_alarm == true) {
      m_alarm = parseInt(m_alarm) - 1;
      if (m_alarm < 0) {
        m_alarm = 59;
        h_alarm = h_alarm - 1;
        if (h_alarm < 1) {
          h_alarm = 12;
          if (session_alarm == "AM") {
            session_alarm = "PM";
          } else {
            session_alarm = "AM";
          }
        }
      }
      m_alarm = m_alarm.toString();
      h_alarm = h_alarm.toString();
      showTime();
      setClock();
    }
}

function increaseSeconds() {
    if (setting_alarm == true) {
      s_alarm = parseInt(s_alarm) + 1;
      if (s_alarm > 59) {
        s_alarm = 0;
        m_alarm = parseInt(m_alarm) + 1;
        if (m_alarm > 59) {
          m_alarm = 0;
          h_alarm = h_alarm + 1;
          if (h_alarm > 12) {
            h_alarm = 1;
            if (session_alarm == "AM") {
              session_alarm = "PM";
            } else {
              session_alarm = "AM";
            }
          }
        }
      }

      m_alarm = m_alarm.toString();
      h_alarm = h_alarm.toString();
      s_alarm = s_alarm.toString();
      showTime();
      setClock();
    }
}

function decreaseSeconds() {
    if (setting_alarm == true) {
      s_alarm = parseInt(s_alarm) - 1;
      if (s_alarm < 0) {
        s_alarm = 59;
        m_alarm = parseInt(m_alarm) - 1;
        if (m_alarm < 0) {
          m_alarm = 59;
          h_alarm = h_alarm - 1;
          if (h_alarm < 1) {
            h_alarm = 12;
            if (session_alarm == "AM") {
              session_alarm = "PM";
            } else {
              session_alarm = "AM";
            }
          }
        }
      }

      m_alarm = m_alarm.toString();
      h_alarm = h_alarm.toString();
      s_alarm = s_alarm.toString();
      showTime();
      setClock();
    }
}

function snooze() {
    if (alarm == true) {
        sound.pause();
        sound.currentTime = 0;
        increaseMinutes();
        increaseMinutes();
        increaseMinutes();
        increaseMinutes();
        increaseMinutes();

    }
}

function AlarmOff() {
  if (activeAlarm == true) {
    activeAlarm = false;
    if (alarm == false) {
      sound.pause();
      sound.currentTime = 0;
    }
    var row = document.getElementsByClassName('alarm alarm_marker')
    for ( var i=0; i<row.length; i++ ) {
      row[i].style.backgroundColor = '#67000D';
    }

  }
}
