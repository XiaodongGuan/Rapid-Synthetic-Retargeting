using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class FS_Hospital : MonoBehaviour
{
    private Object[] person = new Object[5];
    private Transform[] meshTransform = new Transform[5];
    private Object[] cloth = new Object[5];
    private Object[] wig = new Object[5];
    private Object[] glass = new Object[5]; // glasses
    private Object[] sheet_object = new Object[5]; // sheet
    private Texture2D[] tex_person = new Texture2D[5];
    private Texture2D[] tex_cloth = new Texture2D[5];
    private Texture2D[] tex_wig = new Texture2D[5];
    private Texture2D[] tex_glass = new Texture2D[5];  // glasses
    private Texture2D[] tex_sheet = new Texture2D[5];  // sheet
    private GameObject[] mannequin = new GameObject[5];
    private GameObject[] coat = new GameObject[5];
    private GameObject[] hair = new GameObject[5];
    private GameObject[] glasses = new GameObject[5];  // glasses
    private GameObject[] sheet = new GameObject[5];  // sheet
    private Vector3 light_cam_trans;
    private Texture2D nrm_male;
    private Texture2D nrm_female;
    private int picWidth;
    private int picHeight;
    private Vector3 JpoCam;
    private Vector3[,] JpoWorld =  new Vector3[5, 29];
    public GameObject cam;
    public float fl;
    private int img_count;
    private string head;
    private string img_name;
    private int scene;
    private int glasses_en; // glasses
    private int sheet_en;  // sheet
    private int itr;
    private bool mesh_used;
    private TextAsset meshinfo;
    private TextAsset dataset;
    private bool in_operation;
    private bool load_permit;
    private float cx;
    private float cy;
    private float ix;
    private float iy;
    private float y_compensation;

     string dataset_path = "path/to/save_dir/";

    private StreamWriter writer;
    private Vector2 PixPerMM;
    private int gender;
    private int mesh_no;
    // Start is called before the first frame update
    private StreamWriter stat_report;
    private TextAsset stat_txt;
    private int stat_int;
    private int person_count;
    private Vector3 obj_init_trans = new Vector3(0f, 0f, 0f);
    private Camera _camera;
    // Start is called before the first frame update
    public Light[] directionalLights;  // Drag your 5 Directional Light objects here in the Unity Editor
    public float minIntensity = 0.3f;
    public float maxIntensity = 3f;


    void Start()
    {
        img_count = 0; // for indexing the images, can choose arbitrary start number
        itr = 16;   // to speed up generation
        _camera = GetComponent<Camera>();
        _camera.focalLength = fl;
        picHeight = GetComponent<Camera>().pixelHeight;
        picWidth = GetComponent<Camera>().pixelWidth;
        PixPerMM = new Vector2(picWidth / _camera.sensorSize.x, picHeight / _camera.sensorSize.y);
        cx = picWidth * 0.5f;    // optical centre x >
        cy = picHeight * 0.5f;   // optical centre y ^
        ix = 0f;
        iy = 0f;
        transform.position = new Vector3(0f, 0f, 0f);
        var Jpo3DfilePath = "/path/to/your/groundtruth_annotation.txt";
        writer = new StreamWriter(Jpo3DfilePath, false);
        StartCoroutine("AssetRefresh"); 
        nrm_female = Resources.Load<Texture2D>("smplx_texture_f_nrm");
        nrm_male = Resources.Load<Texture2D>("smplx_texture_m_nrm");

    }

    IEnumerator AssetRefresh()
    {
        while(true){
            AssetDatabase.Refresh();
            stat_txt = Resources.Load<TextAsset>("status");  // contains file_ready status and the number of person meshes
            var stat_lines = stat_txt.text.Split('\n');
            var stat_str = stat_lines[0];
            // Debug.Log(stat_str);
            stat_int = int.Parse(stat_str);
            // Debug.Log(stat_int);
            if (stat_int == 1)
            {
                Debug.Log("status check");
                person_count = int.Parse(stat_lines[1]);
                AssetDatabase.Refresh();
                meshinfo = Resources.Load<TextAsset>("mesh_info");
                var m_lines = meshinfo.text.Split('\n');  // number of m_lines' rows equals to person_count
                for (int g = 0; g < person_count; g++)
                {
                    var m_row = m_lines[g].Split(' ');
                    gender = int.Parse(m_row[0]);
                    mesh_no = int.Parse(m_row[1]);
                    scene = int.Parse(m_row[2]);
                    glasses_en = int.Parse(m_row[3]); // glasses
                    sheet_en = int.Parse(m_row[4]); // sheet
                    dataset = Resources.Load<TextAsset>("Jpo" + mesh_no.ToString());
                    var lines = dataset.text.Split('\n');
                    if (lines.Length != 30)
                    {
                        Debug.Log("Bad Jpo Entry");   // there will be an extra line because of searching for '\n'
                        Debug.Log(lines.Length);
                    }
                    for(int i = 0; i < 29; i++) 
                    {
                        var data = lines[i].Split(' ');
                        var list = new List<string>(data); // turn this into a list
                        JpoWorld[g, i] = new Vector3(float.Parse(list[0]), float.Parse(list[1]), float.Parse(list[2]));
                        // Debug.Log(JpoWorld[g, i]);
                    }
                    person[g] = Resources.Load("body" + mesh_no.ToString(), typeof(GameObject));
                    mannequin[g] = Instantiate(person[g], obj_init_trans, Quaternion.identity) as GameObject;
                    meshTransform[g] = mannequin[g].GetComponent<Transform>();
                    tex_person[g] = Resources.Load<Texture2D>("full_body_uv" + mesh_no.ToString());
                    mannequin[g].GetComponentInChildren<Renderer>().material.mainTexture = tex_person[g];
                    mannequin[g].GetComponentInChildren<Renderer>().material.EnableKeyword("_NORMALMAP");
                    mannequin[g].GetComponentInChildren<Renderer>().material.SetFloat("_Smoothness", 0f);
                    if (gender == 0)
                        {mannequin[g].GetComponentInChildren<Renderer>().material.SetTexture("_NormalMap", nrm_male);}

                    else
                        {mannequin[g].GetComponentInChildren<Renderer>().material.SetTexture("_NormalMap", nrm_female);}   // try _BumpMap if nor working
                    switch (scene)
                    {
                        case 1:
                            Loader_1(mesh_no);
                            break;
                        case 2:
                            Loader_2(mesh_no);
                            break;
                        case 3:
                            Loader_3(mesh_no);
                            break;
                        default:
                            break;
                    }
                    G_loader(glasses_en, mesh_no); // glasses
                    S_loader(sheet_en, mesh_no);  //sheet
                }
                Camera_control();

                yield return new WaitForSeconds(0.1f);
            }
            else
            {
                yield return new WaitForSeconds(2f);
            }
        }
    }

    void Loader_1(int midx)
    {
        wig[midx] = Resources.Load("wig" + midx.ToString(), typeof(GameObject));
        hair[midx] = Instantiate(wig[midx], obj_init_trans, Quaternion.identity) as GameObject;
        tex_wig[midx] = Resources.Load<Texture2D>("wig_uv" + midx.ToString());
        hair[midx].GetComponentInChildren<Renderer>().material.mainTexture = tex_wig[midx];
        hair[midx].GetComponentInChildren<Renderer>().material.SetFloat("_Smoothness", 0f);
    }

    void Loader_2(int midx)
    {
        cloth[midx] = Resources.Load("cloth" + midx.ToString(), typeof(GameObject));
        coat[midx] = Instantiate(cloth[midx], obj_init_trans, Quaternion.identity) as GameObject;
        tex_cloth[midx] = Resources.Load<Texture2D>("cloth_uv" + midx.ToString());
        coat[midx].GetComponentInChildren<Renderer>().material.mainTexture = tex_cloth[midx];
        coat[midx].GetComponentInChildren<Renderer>().material.SetFloat("_Smoothness", 0f);
    }

    void Loader_3(int midx)
    {
        Loader_1(midx);
        Loader_2(midx);
    }

    void G_loader(int glasses_En, int midx)  // glasses
    {
        if (glasses_En == 1)
        {
            glass[midx] = Resources.Load("glasses" + midx.ToString(), typeof(GameObject));
            glasses[midx] = Instantiate(glass[midx], obj_init_trans, Quaternion.identity) as GameObject;
            tex_glass[midx] = Resources.Load<Texture2D>("glasses_uv" + midx.ToString());
            glasses[midx].GetComponentInChildren<Renderer>().material.mainTexture = tex_glass[midx];
            glasses[midx].GetComponentInChildren<Renderer>().material.SetFloat("_Smoothness", 0f);
        }
    }

    void S_loader(int sheet_En, int midx)   // sheet
    {
        if (sheet_En == 1)
        {
            sheet_object[midx] = Resources.Load("sheet" + midx.ToString(), typeof(GameObject));
            sheet[midx] = Instantiate(sheet_object[midx], obj_init_trans, Quaternion.identity) as GameObject;
            tex_sheet[midx] = Resources.Load<Texture2D>("sheet_uv" + midx.ToString());
            sheet[midx].GetComponentInChildren<Renderer>().material.mainTexture = tex_sheet[midx];
            sheet[midx].GetComponentInChildren<Renderer>().material.SetFloat("_Smoothness", 0f);
        }
    }

    void Camera_control()   // moving camera, steady lighting direction
    {
        for (int g=0; g<itr; g++)
        {
            // adjust lights
            for (int i = 0; i < directionalLights.Length; i++)
            {
                directionalLights[i].intensity = Random.Range(minIntensity, maxIntensity);
            }

            // relocate camera
            float r = Random.Range(3f, 6f);
            float phi = Random.Range(-0.3f*Mathf.PI, 0.1f*Mathf.PI);
            float theta = 0f;
            theta = Random.Range(-0.5f*Mathf.PI, 0.5f*Mathf.PI);
            _camera.transform.position = new Vector3(r*Mathf.Cos(phi)*Mathf.Cos(theta), r*Mathf.Sin(phi), r*Mathf.Cos(phi)*Mathf.Sin(theta));
            _camera.transform.LookAt(obj_init_trans);


            float z_rot = 0f;
            // rotate the camera
            z_rot = Random.Range(-115f, -30f);
            
            _camera.transform.Rotate(new Vector3(0, 0, z_rot));
            Debug.Log(string.Format("Camera position: {0}", _camera.transform.position));
            Debug.Log(string.Format("Camera rotation: {0}", _camera.transform.rotation));

            var m = _camera.worldToCameraMatrix;     // renew matrix for WORLD 3D --> CAM 3D   _camera.WorldToScreenPoint
            img_count ++;

            // RGB rendering and saving
            img_name = "SPED_X_CROWDED_" + img_count.ToString() + ".png";
            RenderTexture rt = new RenderTexture(picWidth, picHeight, 24);
            _camera.targetTexture = rt;
            Texture2D screenShot = new Texture2D(picWidth, picHeight, TextureFormat.RGB24, false);
            _camera.Render();
            RenderTexture.active = rt;
            screenShot.ReadPixels(new Rect(0, 0, picWidth, picHeight), 0, 0);
            _camera.targetTexture = null;
            RenderTexture.active = null; // JC: added to avoid errors
            Destroy(rt);
            byte[] bytes = screenShot.EncodeToPNG();
            System.IO.File.WriteAllBytes(dataset_path + img_name, bytes);
            bytes = null;

            for (int mid = 0; mid < person_count; mid++)
            {
                for(int i = 0; i < 29; i++)
                {
                    Vector3 localPoint = JpoWorld[mid, i];
                    // Convert the local point to world space
                    Vector3 worldPoint = meshTransform[mid].TransformPoint(localPoint);
                    // Convert the world point to camera space
                    Vector3 cameraPoint = _camera.transform.InverseTransformPoint(worldPoint);
                    // Convert the world point to screen
                    Vector3 screenPoint = _camera.WorldToScreenPoint(worldPoint);  // x(width coordinate on screen, 0 on left), y(height coordinate on screen, 0 on bot), z(camera space depth)
                    StringBuilder sb = new StringBuilder("v ", 100);
                    sb.Append(cameraPoint.x.ToString()).Append(' ').Append(cameraPoint.y.ToString()).Append(' ').Append(cameraPoint.z.ToString()).Append(' ').Append(screenPoint.x.ToString()).Append(' ').Append(screenPoint.y.ToString());
                    var content = sb.ToString();
                    writer.WriteLine(content);
                }
            }
            writer.WriteLine("f " + img_name);
            Debug.Log(string.Format("Took screenshot to: {0}", dataset_path + img_name));
            stat_report = new StreamWriter("/path/to/your/UnityProjects/Assets/Resources/status.txt", false);
            stat_report.WriteLine("0");
            // Debug.Log("0");
            // AssetDatabase.Refresh();
            stat_report.Close();
        }
        for (int mid = 0; mid < person_count; mid++)
        {
            Destroy(mannequin[mid]);
            Destroy(coat[mid]);
            Destroy(hair[mid]);
            Destroy(glasses[mid]);  // glasses
            Destroy(sheet[mid]);  // sheet
        }

        for (int mid = 0; mid < person_count; mid++)   // avoid accumulating
        {
            Resources.UnloadAsset(stat_txt);
            Resources.UnloadAsset(meshinfo);
            Resources.UnloadAsset(dataset);
            person[mid] = null;
            Resources.UnloadAsset(tex_person[mid]);
            wig[mid] = null;
            Resources.UnloadAsset(tex_wig[mid]);
            Resources.UnloadAsset(cloth[mid]);
            Resources.UnloadAsset(tex_cloth[mid]);
            glass[mid] = null;
            Resources.UnloadAsset(tex_glass[mid]);
            Resources.UnloadAsset(sheet_object[mid]);
            Resources.UnloadAsset(tex_sheet[mid]);
        }
        
        Resources.UnloadUnusedAssets();
        System.GC.Collect();
    }


    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey("q") || img_count >= 100001)
        {
            writer.Close();
            stat_report.Close();
            StopCoroutine("AssetRefresh");
            Debug.Log("safe to quit");
        }
    }
}
