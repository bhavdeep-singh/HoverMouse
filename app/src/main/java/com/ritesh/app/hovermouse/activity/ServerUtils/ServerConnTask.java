package com.ritesh.app.hovermouse.activity.ServerUtils;

import android.content.Context;
import android.os.AsyncTask;
import android.provider.SyncStateContract;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;

/**
 * Created by ritesh_kumar on 04-Mar-16.
 */
public class ServerConnTask extends AsyncTask<String, Void, Boolean> {

    private Context mContext;
    private Socket socket;

    public ServerConnTask(Context context) {
        mContext = context;
    }

    @Override
    protected Boolean doInBackground(String... params) {
        boolean result = true;
        try {
            InetAddress serverAddr = InetAddress.getByName(params[0]);
            socket = new Socket(serverAddr, Constants.SERVER_PORT);//Open socket on server IP and port
        } catch (IOException e) {
            Log.e("remotedroid", "Error while connecting", e);
            result = false;
        }
        return result;
    }

    /*@Override
    protected void onPostExecute(Boolean result) {
        isConnected = result;
        Toast.makeText(context, isConnected ? "Connected to server!" : "Error while connecting", Toast.LENGTH_LONG).show();
        try {
            if (isConnected) {
                out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(socket
                        .getOutputStream())), true); //create output stream to send data to server
            }
        } catch (IOException e) {
            Log.e("remotedroid", "Error while creating OutWriter", e);
            Toast.makeText(context, "Error while connecting", Toast.LENGTH_LONG).show();
        }
    }*/
}