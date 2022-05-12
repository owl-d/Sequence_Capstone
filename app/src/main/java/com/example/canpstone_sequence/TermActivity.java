package com.example.canpstone_sequence;

import static com.example.canpstone_sequence.ETRIActivity.bills_mode;
import static com.example.canpstone_sequence.ETRIActivity.record_end;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class TermActivity extends AppCompatActivity {

    Button btn_bills;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_term);

        btn_bills = (Button) findViewById(R.id.btn_bills);
        if (bills_mode == 2) {
            btn_bills.setText("상품 찾기가 끝났습니다!");
        }
        btn_bills.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bills_mode == 2) {
                    btn_bills.setText("상품 찾기가 끝났습니다!");
                }
                else {
                    record_end = false;
                    bills_mode = 1;
                    Intent intent = new Intent(getApplicationContext(), CameraActivity.class);
                    startActivity(intent);
                }
            }
        });
    }
}