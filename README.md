# Hover Mouse

Hover Mouse is android application which converts your mobile device to a wireless mouse. The special feature of this app is that you can operate the mouse in air i.e. no need of plane surface, unlike the conventional optical mouse.

**Server for Laptop** : [HoverMouseServer](https://github.com/uneq95/HoverMouseServer)

##### What is functional now?

 * We can move the mouse pointer (movement lags!)
 * Single and Double left click
 * Right click

### How to use this?

1. Create hotspot from your laptop.
2. If the IPv4 address of the hotspot network is not 192.168.137.1 then change the constant [SERVER_IP](https://github.com/uneq95/HoverMouse/blob/master/app/src/main/java/com/ritesh/app/hovermouse/activity/ServerUtils/Constants.java) and rebuild the project.
3. Open up the [HoverMouse Server](https://github.com/uneq95/HoverMouseServer) and run the [RemoteDroidServer.java](https://github.com/uneq95/HoverMouseServer/blob/master/src/RemoteDroidServer.java) file.
4. Also remember to turn off the windows firewall, if working on Windows.
5. Now open up the android app and press the back button, if it shows a Toast message "Connect to server", then you can notice you have control to the mouse pointer, otherwise try pressing the back button for connecting to server. If still cannot connect, then restart the server again. 

##### Dependencies

* [OpenCV-Android-SDK](http://opencv.org/downloads.html): Import the sdk as module in Android Studio and then in project properties, add it as a dependency.


### Stuff used to make this:

 * [Android App for machine vision](http://barrythomas.co.uk/code.html) for Optical Flow (Thanks to [Barry Thomas](http://uk.linkedin.com/in/thatbarrythomas))
