import torch
from tqdm import tqdm

def data_reshape(y_pred, y_label, y_mask):
    # 입력되는 y_pred : (Batch_size, seq_len, output_dim)
    # 입력되는 y_label : (Batch_size, seq_len)
    batch_size, seq_len, output_dim = y_pred.size()

    re_y_pred = y_pred.view(-1, output_dim) # (bs*seq, out)
    re_y_label = y_label.view(-1) #(bs*seq)
    re_y_mask = y_mask.view(-1) #마스크는 라벨이랑 차원변환과정이 동일하게 수행됨

    return re_y_pred, re_y_label, re_y_mask



# 정답을 맞출 때 '무시'해야 할 클래스가 있을때 동작하는 함수
def cal_correct(y_pred, y_label, mask=None):
    if y_pred.size(1) != 1:
        pred = y_pred.argmax(dim=1) #가장 높은 예측값 하나 추출
    else : #y_pred가 crf레이어로 인해 이미 최적의 예측값인 경우
        pred = y_pred.squeeze(dim=1) #(bs*seq_len, 1)-> #(bs*seq_len)

    if mask is not None: #마스크된 항목이 존재할 때
        correct = pred.eq(y_label).masked_select(mask).sum().item()
        total = mask.sum().item() # 전체 원소 개수중 마스크처리된것만 분모
    else:
        correct = pred.eq(y_label).sum().item()
        total = y_label.numel() # 전체 원소 수를 분모로 처리함
    
    # 수치적 안정성을 보장하면서 연산을 수행하자
    iter_cor = correct / total if total > 0 else 0
    return iter_cor



def model_train(model, data_loader, optimizer_fn, 
                epoch, epoch_step, 
                loss_fn=None, ignore_class=None,
                CRF=False):
    # 1개의 epoch내 batch단위(iter)로 연산되는 값이 저장되는 변수들
    iter_size, iter_loss, iter_correct = 0, 0, 0

    device = next(model.parameters()).device # 모델의 연산위치 확인
    model.train() # 모델을 훈련모드로 설정

    #특정 epoch_step 단위마다 tqdm 진행바가 생성되게 설정
    if (epoch+1) % epoch_step == 0 or epoch == 0:
        tqdm_loader = tqdm(data_loader)
    else:
        tqdm_loader = data_loader

    for x_data, y_label in tqdm_loader:
        x_data, y_label = x_data.to(device), y_label.to(device)
        # y_label을 바탕으로 mask를 생성하기
        y_mask = (y_label != ignore_class)

        if CRF != True : #CRF 레이어가 없을 때
            y_pred = model(x_data) # Forward, 모델이 예측값을 만들게 함
        else : # CRF Forward, 모델이 예측값이랑 loss 둘다 출력함
            y_pred, loss = model(x_data, y_label, y_mask)
            loss = loss.mean() #loss는 (bs)여서 -> 1 스칼라로 바꿔줘야함

        # 데이터의 적절한 구조변환 수행
        y_pred, y_label, y_mask = data_reshape(y_pred, y_label, y_mask)
        
        if loss_fn is not None: # 로스함수가 존재할 경우 해당 코드 실행
            # CRF는 loss 함수가 없으니 이 코드는 넘어가게 된다.
            loss = loss_fn(y_pred, y_label)

        #backward 과정 수행
        optimizer_fn.zero_grad()
        loss.backward()
        optimizer_fn.step() # 마지막에 스케줄러 있으면 업뎃코드넣기

        # 현재 batch 내 샘플 개수당 correct, loss, 수행 샘플 개수 구하기
        iter_correct += cal_correct(y_pred, y_label, 
                                    y_mask) * x_data.size(0)
        iter_loss += loss.item() * x_data.size(0)
        iter_size += x_data.size(0)

        # tqdm에 현재 진행상태를 출력하기 위한 코드
        if (epoch+1) % epoch_step == 0 or epoch == 0:
            prograss_loss = iter_loss / iter_size
            prograss_acc = iter_correct / iter_size
            desc = (f"[훈련중]로스: {prograss_loss:.3f}, "
                    f"정확도: {prograss_acc:.3f}")
            tqdm_loader.set_description(desc)

    #현재 epoch에 대한 종합적인 정확도/로스 계산
    epoch_acc = iter_correct / iter_size
    epoch_loss = iter_loss / len(data_loader.dataset)
    return epoch_loss, epoch_acc


def model_evaluate(model, data_loader,
                    epoch, epoch_step, 
                    loss_fn=None, ignore_class=None,
                    CRF=False):
    # 1개의 epoch내 batch단위(iter)로 연산되는 값이 저장되는 변수들
    iter_size, iter_loss, iter_correct = 0, 0, 0

    device = next(model.parameters()).device # 모델의 연산위치 확인
    model.eval() # 모델을 평가 모드로 설정

    #특정 epoch_step 단위마다 tqdm 진행바가 생성되게 설정
    if (epoch+1) % epoch_step == 0 or epoch == 0:
        tqdm_loader = tqdm(data_loader)
    else:
        tqdm_loader = data_loader

    with torch.no_grad(): #평가모드에서는 그래디언트 계산 중단
        for x_data, y_label in tqdm_loader:
            x_data, y_label = x_data.to(device), y_label.to(device)

            # y_label을 바탕으로 mask를 생성하기
            y_mask = (y_label != ignore_class)

            if CRF != True : #CRF 레이어가 없을 때
                y_pred = model(x_data) # Forward, 모델이 예측값을 만들게 함
            else : # CRF Forward, 모델이 예측값이랑 loss 둘다 출력함
                y_pred, loss = model(x_data, y_label, y_mask)
                loss = loss.mean() #loss는 (bs)여서 -> 1 스칼라로 바꿔줘야함

            # 데이터의 적절한 구조변환 수행
            y_pred, y_label, y_mask = data_reshape(y_pred, y_label, y_mask)

            if loss_fn is not None: # 로스함수가 존재할 경우 해당 코드 실행
                # CRF는 loss 함수가 없으니 이 코드는 넘어가게 된다.
                loss = loss_fn(y_pred, y_label)

            # 현재 batch 내 샘플 개수당 correct, loss, 수행 샘플 개수 구하기
            iter_correct += cal_correct(y_pred, y_label, 
                                        y_mask) * x_data.size(0)
            iter_loss += loss.item() * x_data.size(0)
            iter_size += x_data.size(0)

    #현재 epoch에 대한 종합적인 정확도/로스 계산
    epoch_acc = iter_correct / iter_size
    epoch_loss = iter_loss / len(data_loader.dataset)
    return epoch_loss, epoch_acc